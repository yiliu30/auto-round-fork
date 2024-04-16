import matplotlib.pyplot as plt

# Sample log data, replace this with the path to your log file
log_data_file = "_fl_llama-2-7b-4bits-iter5000-log_forparse.txt"

# Parse the log data
iterations = []
losses = []
learning_rates = []

start_flag = 0
target_block = 32
with open(log_data_file, 'r') as f:
    log_data_all = f.readlines()
    # Split the log data into lines and process each line
    for line in log_data_all:
        if "quantizing block" in line:
            print(line)
            start_flag += 1
            if start_flag > target_block: 
                break
        if start_flag == target_block:
            print(f"block {target_block} found")
            line = line.strip("\n")
            if 'loss:' in line:
                
                parts = line.split()
                iteration = int(parts[4].strip(','))
                loss = float(parts[6])
                iterations.append(iteration)
                losses.append(loss)
            if 'lr is' in line:
                parts = line.split()
                iteration = int(parts[4].strip(','))
                lr = float(parts[-1].strip('[]'))
                learning_rates.append((iteration, lr))

# Prepare data for plotting
lr_iterations, lrs = zip(*learning_rates)  # unzip list of tuples

# Plotting
plt.title(f'Loss and Learning Rate per Iteration - {target_block}th block ')
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(iterations, losses, marker='o', color='red')
plt.title(f'Loss per Iteration - {target_block}th block')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(lr_iterations, lrs, marker='o', color='blue')
plt.title(f'Learning Rate per Iteration - {target_block}th block')
plt.xlabel('Iteration')
plt.ylabel('Learning Rate')

plt.tight_layout()
plt.savefig(f'loss_lr_plot+{target_block}.png')
