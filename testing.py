import requests
import time
import pandas as pd
import matplotlib.pyplot as plt

BASE_URL = 'http://aimodel-env.eba-dpi6ummi.us-east-2.elasticbeanstalk.com/predict'

# Test cases to check the model predictions
TEST_CASES = [
    {"text": "this year is 2030"},  
    {"text": "I living in sun!"},  
    {"text": "UOFT is one of best universities in the world"}, 
    {"text": "AI is growing fast"}  


latency_data = []


def send_request(text):
    """Send POST request to the API and return the prediction and response."""
    try:
        response = requests.post(BASE_URL, json={"text": text})
        print(f"Request: {text} | Status Code: {response.status_code} | Response: {response.text}")

        if response.status_code == 200:
            prediction = response.json().get('prediction')
            return prediction, response.text  # Return prediction and response text
        else:
            return None, response.text  
    except Exception as e:
        print(f"Error occurred during the request: {str(e)}")
        return None, str(e) 

# Function to run functional/unit tests
def run_functional_tests():
    """Run functional tests and print predictions."""
    print("Running Functional/Unit tests:")
    for case in TEST_CASES:
        prediction, response_text = send_request(case['text'])
        print(f"Input: {case['text']} => Prediction: {prediction}")

# Function to perform latency and performance testing
def run_latency_tests():
    """Measure latency for each test case and collect results."""
    print("\nRunning Latency/Performance tests:")
    for case in TEST_CASES:
        text = case['text']
        for _ in range(100):  # 100 API calls
            start_time = time.time()
            prediction, response_text = send_request(text)
            latency = time.time() - start_time
            latency_data.append({
                "text": text,
                "latency": latency,
                "response": response_text
            })

# Function to save latency data to a CSV file
def save_latency_to_csv(filename='latency_results_with_responses.csv'):
    """Save latency data to a CSV file."""
    df_latency = pd.DataFrame(latency_data)
    df_latency.to_csv(filename, index=False)
    print(f"\nLatency results saved to {filename}")

# Function to generate a boxplot for latency data
def generate_boxplot(filename='latency_boxplot.png'):
    """Generate and save a boxplot for the latency data."""
    df_latency = pd.DataFrame(latency_data)
    plt.figure(figsize=(10, 6))
    df_latency.boxplot(column='latency', by='text')
    plt.title('Latency Performance Boxplot')
    plt.suptitle('')
    plt.xlabel('Test Case')
    plt.ylabel('Latency (seconds)')
    plt.grid()
    plt.savefig(filename)
    print(f"Boxplot saved as '{filename}'")

# Function to calculate and print average latency per test case
def calculate_average_latency():
    """Calculate and print average latency for each test case."""
    df_latency = pd.DataFrame(latency_data)
    average_latency = df_latency.groupby('text')['latency'].mean()
    print("\nAverage Latency (seconds):")
    print(average_latency)
    return average_latency  # Return in case further use is needed


def main():
    """Main function to run the tests and generate reports."""
    run_functional_tests()           # Run functional/unit tests
    run_latency_tests()              # Measure latency
    save_latency_to_csv()            # Save results to CSV
    generate_boxplot()               # Generate and save boxplot
    calculate_average_latency()      # Print average latency

if __name__ == '__main__':
    main()
