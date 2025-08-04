import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Regressor Neural Network
class RegressorNet(nn.Module):
	def __init__(self, input_size=9):
		super(RegressorNet, self).__init__()
		self.model = nn.Sequential(
			nn.Linear(input_size, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, 32),
			nn.ReLU(),
			nn.Linear(32, 32),
			nn.ReLU(),
			nn.Linear(32, 16),
			nn.ReLU(),
			nn.Linear(16, 1)
		)
	def forward(self, x):
		return self.model(x)

# Function to load data
def load_data():
	nb_data = 2800
	w_data = np.zeros((nb_data, 9))
	w_labels = np.zeros(nb_data)

	ra = list(range(1000, 3800))

	cpt = 0
	for i in ra:
		dat = np.load(f'../../../experiments/Coinrun/vector/data_vector_optim_{i}.npy')
		w_data[cpt, :] = dat[:9]
		val = np.load(f'../../../experiments/Coinrun/generalization/Average_reward{i}.npy')
		w_labels[cpt] = val[0][0]
		cpt += 1
	#w_labels=w_labels/10
	return w_data, w_labels

# Train the network
def train_model(model, criterion, optimizer, x_train, y_train, x_eval, y_eval, epochs=200000, patience=200000):
	best_loss = np.inf
	patience_counter = 0

	for epoch in range(epochs):
		model.train()
		optimizer.zero_grad()
		outputs = model(x_train)
		loss = criterion(outputs, y_train)
		loss.backward()
		optimizer.step()

		model.eval()
		with torch.no_grad():
			val_outputs = model(x_eval)
			val_loss = criterion(val_outputs, y_eval)

		if val_loss.item() < best_loss:
			best_loss = val_loss.item()
			patience_counter = 0
			best_model_state = model.state_dict()
		else:
			patience_counter += 1

		if patience_counter >= patience:
			break

	model.load_state_dict(best_model_state),
	return model

def plot_predictions(predictions,labels,filename, title):
	sorted_indices = np.argsort(labels)
	sorted_preds = predictions[sorted_indices]
	sorted_labels = labels[sorted_indices]
	x_ind = np.arange(len(labels))
	
	plt.scatter(x_ind, sorted_labels, zorder=1, s=5, label='Actual')
	plt.scatter(x_ind, sorted_preds, zorder=2, s=5, label='Predicted')
	plt.title(title)
	plt.legend()
	plt.savefig(filename)
	plt.clf()

# Main training and evaluation function
def generate_metrics():
	w_data, w_labels = load_data()

	# Convert to tensors
	w_data_tensor = torch.tensor(w_data, dtype=torch.float32).to(device)
	w_labels_tensor = torch.tensor(w_labels, dtype=torch.float32).unsqueeze(1).to(device)

	# Split datasets
	w_data_train, w_labels_train = w_data_tensor[:-300], w_labels_tensor[:-300]
	w_data_eval, w_labels_eval = w_data_tensor[-300:-150], w_labels_tensor[-300:-150]
	w_data_test, w_labels_test = w_data_tensor[-150:], w_labels_tensor[-150:]

	# Instantiate models
	regressor = RegressorNet().to(device)
	# Define loss and optimizers
	criterion_reg = nn.MSELoss()
	optimizer_reg = optim.Adam(regressor.parameters(), lr=0.0001)
	# Train models
	for it_tr in range(0,200):
		regressor = train_model(regressor, criterion_reg, optimizer_reg, w_data_train, w_labels_train, w_data_eval, w_labels_eval,epochs=30000,patience=30000)
		# Save models
		torch.save(regressor.state_dict(), '../../../experiments/Coinrun/models/vector_regressor.pth')
	# Predictions
		regressor.eval()
		with torch.no_grad():
			preds_reg_test = regressor(w_data_test).cpu().numpy().flatten()
			preds_reg_train = regressor(w_data_train).cpu().numpy().flatten()
		preds_reg_test[preds_reg_test<0]=0
		preds_reg_train[preds_reg_train<0]=0
		# Plot predictions for test data
		plot_predictions(preds_reg_test, w_labels_test.cpu().numpy().flatten(), 'TEST-Generalization_prediction_vector_pytorch_'+str(it_tr)+'.pdf', 'Generalization (Regression) - Test')
		# Plot predictions for training data
		plot_predictions(preds_reg_train, w_labels_train.cpu().numpy().flatten(), 'TRAIN-Generalization_prediction_vector_pytorch_'+str(it_tr)+'.pdf', 'Generalization (Regression) - Train')

def main():
	random.seed(123456)
	np.random.seed(123456)
	torch.manual_seed(123456)
	
	if not os.path.exists('../../../experiments/Coinrun/models'):
		os.makedirs('../../../experiments/Coinrun/models')

	generate_metrics()

if __name__ == "__main__":
	main()
