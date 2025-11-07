from fraud_pipeline import fraud_detection_pipeline
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

# Create deployment
deployment = Deployment.build_from_flow(
    flow=fraud_detection_pipeline,
    name="fraud-detection-daily",
    schedule=CronSchedule(cron="0 2 * * *"),  # Run daily at 2 AM
    work_pool_name="default",
    tags=["production", "fraud-detection", "ml"],
)

if __name__ == "__main__":
    deployment.apply()
    print("✅ Deployment created!")
    print("🌐 View at: http://localhost:4200/deployments")