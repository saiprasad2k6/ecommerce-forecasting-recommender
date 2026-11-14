"""RustFS readiness retries transient failures without hiding credential errors."""
import unittest
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError, EndpointConnectionError
from scripts.storage.start_rustfs import wait_for_storage


def service_error(status):
    return ClientError({"Error": {"Code": str(status)},
                        "ResponseMetadata": {"HTTPStatusCode": status}}, "ListBuckets")


class StorageStartupTests(unittest.TestCase):
    @patch("scripts.storage.start_rustfs.time.sleep")
    def test_retries_startup_errors_until_ready(self, sleep):
        for error in [service_error(503), EndpointConnectionError(endpoint_url="http://localhost:19000")]:
            with self.subTest(error=type(error).__name__):
                client = Mock()
                client.list_buckets.side_effect = [error, {"Buckets": []}]
                sleep.reset_mock()
                wait_for_storage(client, attempts=2)
                self.assertEqual(client.list_buckets.call_count, 2)
                sleep.assert_called_once_with(1)

    @patch("scripts.storage.start_rustfs.time.sleep")
    def test_credentials_errors_fail_immediately(self, sleep):
        client = Mock()
        error = service_error(403)
        client.list_buckets.side_effect = error
        with self.assertRaises(ClientError) as raised:
            wait_for_storage(client)
        self.assertIs(raised.exception, error)
        client.list_buckets.assert_called_once()
        sleep.assert_not_called()

    @patch("scripts.storage.start_rustfs.time.sleep")
    def test_persistent_failure_stops_with_original_cause(self, sleep):
        client = Mock()
        error = service_error(503)
        client.list_buckets.side_effect = error
        with self.assertRaisesRegex(RuntimeError, "container logs") as raised:
            wait_for_storage(client, attempts=2)
        self.assertIs(raised.exception.__cause__, error)
        self.assertEqual(client.list_buckets.call_count, 2)
        sleep.assert_called_once_with(1)
