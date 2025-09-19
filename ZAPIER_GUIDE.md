# Zapier Integration Guide

## Step 1: Trigger
- Choose Schedule or Google Sheets row.

## Step 2: Webhook
- Action: POST
- URL: http://YOUR_SERVER:8000/ask
- Body: {"question":"Explain the benefits of sleep."}

## Step 3: Parse JSON
Use `answer.english`, `answer.spanish`, `answer.citations`.

## Step 4: Publish
Example: WordPress -> Create Post
