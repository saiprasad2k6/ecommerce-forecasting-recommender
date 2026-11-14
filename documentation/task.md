# MLOps System Design: Demand Forecasting and Recommendations

## Problem Statement

You are working as an ML Engineer for a leading retail e-commerce platform in the country. This platform caters to millions of customers by offering a wide range of fashion products. Known for its innovative approach, the company has been a market leader, but recent challenges have surfaced.

Despite being a market leader, the company is losing regular customers to competitors due to:

1. **Outdated recommendation system:** Customers find the current recommendation system outdated and irrelevant to their evolving fashion preferences.
2. **Insufficient deals and offers:** Competitive discounts and personalised promotions are lacking, which is pushing customers to rival platforms.
3. **Inefficient demand management:** The platform struggles to handle sudden surges in demand during festivals and seasonal events.

This leads to system inefficiencies and poor user experiences, as the existing infrastructure is not optimised to predict and manage demand spikes.

## Business Objective

To address these challenges, the company wants to revamp its system to deliver better recommendations, handle demand surges efficiently, and improve overall platform performance.

The business requires solutions capable of:

1. **Enhancing customer experience**
	- Deliver personalised, contextually relevant, and up-to-date recommendations based on customer behaviour and fashion trends.
	- Offer tailored deals and discounts to retain customer loyalty.

2. **Demand prediction and scalability**
	- Forecast and handle demand surges during festivals and seasons efficiently.
	- Ensure that the platform remains responsive and operational even under high traffic loads.

## Assignment Task

You must create an ML system that has the features of a complete production stack, from experiment tracking to automated model deployment and monitoring. The framework should also be able to train if additional annotations are available later.

Your solution should answer the following questions:

1. **System design:** Based on the above information, describe the KPIs that the business should track.
2. **MLOps advantages:** Your company has decided to build an MLOps system. What advantages would you get from building an MLOps system rather than a simple model?
3. **System architecture:** Create an ML system design diagram with features of a complete production stack, from experiment tracking to automated model deployment and monitoring.

	The MLOps tools you use are up to your judgment. You may use open-source tools, managed service tools, or a hybrid approach. You may use Draw.io or another tool to create the architecture diagram.

4. **Tool selection justification:** Explain why you chose each tool in the architecture design.
5. **Workflow of the solution:** Specify the steps required to build the system end to end. The steps should mention the tools used in each component and explain how they connect to solve the problem.

	The workflow should broadly include:

	- Data and model experimentation
	- Automation of the data pipeline
	- Automation of the training pipeline
	- Automation of the inference pipeline
	- Continuous monitoring pipeline

## Required Response Scenarios

The workflow should also explain the actions to take under the following conditions.

### Data Drift Handling

Explain which component or pipeline will be triggered if drift is detected. What should happen if the drift exceeds an acceptable threshold?

### Additional Data Handling

Explain which component or pipeline will be triggered when additional annotated data becomes available.