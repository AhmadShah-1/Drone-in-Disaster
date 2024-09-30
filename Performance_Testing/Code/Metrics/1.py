import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File path to the CSV generated by the detection script
csv_file_path = "H:/Direct/Research/FIU/DroneDetection/Drone-in-Disaster/Performance_Testing/images/Detections/Combination/yolo_detections.csv"

# Read the CSV data into a pandas DataFrame
data = pd.read_csv(csv_file_path)

# Set up the plotting style
sns.set(style="whitegrid")

# Create a bar chart for the average number of person detections per model
plt.figure(figsize=(10, 6))
avg_detections = data.groupby('Model')['Number of Person Detections'].mean().reset_index()
sns.barplot(x='Model', y='Number of Person Detections', data=avg_detections, palette="viridis")
plt.title('Average Number of Person Detections per Model')
plt.ylabel('Average Number of Person Detections')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("H:/Direct/Research/FIU/DroneDetection/Drone-in-Disaster/Performance_Testing/images/Detections/Combination/avg_detections_per_model.png")
plt.show()

# Create a box plot for the average confidence scores per model
plt.figure(figsize=(10, 6))
sns.boxplot(x='Model', y='Average Confidence Score', data=data, palette="coolwarm")
plt.title('Distribution of Average Confidence Scores per Model')
plt.ylabel('Average Confidence Score')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("H:/Direct/Research/FIU/DroneDetection/Drone-in-Disaster/Performance_Testing/images/Detections/Combination/confidence_scores_per_model.png")
plt.show()

# Create a line plot to show the trend of average confidence scores over images for each model
plt.figure(figsize=(12, 7))
sns.lineplot(x='Image', y='Average Confidence Score', hue='Model', data=data, marker='o')
plt.title('Trend of Average Confidence Scores per Image for Each Model')
plt.ylabel('Average Confidence Score')
plt.xlabel('Image')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("H:/Direct/Research/FIU/DroneDetection/Drone-in-Disaster/Performance_Testing/images/Detections/Combination/confidence_trend_per_image.png")
plt.show()
