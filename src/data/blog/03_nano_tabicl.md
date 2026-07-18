---
author: Brian Staber
pubDatetime: 2026-07-11T12:16:08Z
modDatetime: 2026-07-11T12:16:08Z
title: Implementing a tiny TabICL
slug: tiny-tabicl
featured: false
draft: false
tags:
  - tabular foundation models
  - tabicl
  - transformer
description: Understanding the building blocks of TabICL v2
---

# Introduction
ss
Tabular foundation models (TFMs) are attracting a lot of attention (no pun intended). In this post, I want to understand what happens under the hood of TabICLv2, an open-source TFM developed by the SODA team at Inria. Other models, such as TabPFN from Prior Labs, follow the same broad idea: pretrain a Transformer on millions of synthetic tabular prediction tasks, then use in-context learning to solve a new task without updating the model's weights.

At inference time, the model receives the labeled training rows as context and predicts the labels or target values of unseen rows. The model conditions on an entire training set, but it does not run gradient descent or fit new parameters for that dataset. TabICLv2 supports classification and regression and handles numerical and categorical features, missing values, and outliers.

My goal here is not to reproduce the full system. I will implement a small version of its three-stage architecture: a column encoder, a row encoder, and a dataset-wise Transformer that performs in-context learning. I will then train it on a controlled synthetic prior and test whether it actually learns to make better predictions as the context set grows.

I used the official [NanoTabICL](https://github.com/soda-inria/nanotabicl/) and [TabICL](https://github.com/soda-inria/tabicl) implementations to understand what's going on. 

# Architecture

TDB.

# Implementation

TBD.

# Training this tiny TabICL with my own prior data

TBD.