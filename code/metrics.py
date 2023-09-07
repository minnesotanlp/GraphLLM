
import csv
# computes accuracy based on the no of records in the file
def compute_accuracy(csv_filename):
    total_count = 0
    correct_count = 0
    fail_count = 0
    with open(csv_filename, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            total_count += 1
            ground_truth = row['GroundTruth']
            parsed_value = row['Parsed Value']
            if ground_truth == parsed_value:
                correct_count += 1
            elif ground_truth in parsed_value: # this is probably not correct way
                correct_count +=1
            if parsed_value == '?':
                fail_count+=1

    if total_count == 0:
        return 0
    else:
        accuracy = correct_count / total_count
        failure_perc = fail_count / total_count
        return accuracy, failure_perc

# Function to record metrics in "metrics.csv"
def record_metrics(metrics_filename, hops, use_edge, sampled_nodes, mean_accuracy, std_accuracy, mean_failure, std_failure, mean_token_perc, std_token_perc):
    with open(metrics_filename, 'a') as metrics_file:
        metrics_writer = csv.writer(metrics_file)
        metrics_writer.writerow([hops, use_edge, sampled_nodes, mean_accuracy, std_accuracy, mean_failure, std_failure, mean_token_perc, std_token_perc])

