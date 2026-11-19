import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Hyperparametros
BATCH_SIZE = 64
MAX_EPOCHS = 5
MIN_ACC = .95
LEARNING_RATE = 1e-3

# Usa GPU se disponível
useGpu = torch.cuda.is_available()
device = torch.device("cuda" if useGpu else "cpu")
print("Using GPU" if useGpu else "Using CPU")

# Transformação dos dados
# - Transforma as imagens em tensores [1, 28, 28]
# - Normaliza utilizando média e desvio padrão do MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# Datasets transformados
train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

# Carregadores de dados
# - Treinamento é feito em lotes de acordo com o hyperparametro e os dados são
#   embaralhados para evitar overfitting baseado na posição dos dados
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Modelo
# Imagem (1, 28, 28) -> Digito (10)
class ModelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolução1: (1, 28, 28) -> (32, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # Convolução2: (32, 28, 28) -> (64, 28, 28)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # MaxPool: Fator de redução 2
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Dropout 25%
        self.dropout = nn.Dropout(0.25)

        # Linear1: 64*7*7 -> 128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # Linear2: 128 -> 10
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x (1, 28, 28)
        # Convolução1 -> ReLU -> MaxPool (32, 14, 14)
        # Convolução2 -> ReLU -> MaxPool (64, 7, 7)
        # flatten (3136)
        # Linear1 -> ReLU -> Dropout (128)
        # Linear2 (10)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

model = ModelCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Treino
def train_one_epoch(epoch):
    # Mode de treino: Dropout ativado
    model.train()

    # Valor de erro acumulado ao longo dos lotes
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()              # Limpa o gradiente prévio
        outputs = model(images)            # Calcula a previsão
        loss = criterion(outputs, labels)  # Calcula o erro
        loss.backward()                    # Calcula o gradiente
        optimizer.step()                   # Atualiza os pesos

        running_loss += loss.item()

        # Print do progresso
        if (batch_idx + 1) % 200 == 0:
            print(f"Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] "
                  f"avg loss: {running_loss / 200:.4f}")
            running_loss = 0.0

# Teste
def evaluate():
    # Modo de treino: Dropout desativado
    model.eval()
    correct, total, test_loss = 0, 0, 0.0

    # Dados para matriz de confusão
    all_predics, all_labels = [], []

    # Ignora computação do gradiente com `no_grad()`, já que não será chamado `backward()`.
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Calcula a previsão
            outputs = model(images)

            # Calcula o erro total no dataset de teste
            test_loss += criterion(outputs, labels).item() * labels.size(0)
            # Índices dos dígitos com maior pontuação em cada imagem
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_predics.extend(predictions.cpu())
            all_labels.extend(labels)

    accuracy = 100.0 * correct / total
    print(f"Test loss: {test_loss / total:.4f} | Test accuracy: {accuracy:.2f}%")
    return accuracy, all_predics, all_labels

# Loop principal
# Passa por MAX_EPOCHS épocas ou até a acurácia for maior que MIN_ACC
predics, labels = [], []
for epoch in range(1, MAX_EPOCHS + 1):
    train_one_epoch(epoch)
    acc, predics, labels = evaluate()

    if acc > 100*MIN_ACC:
        break

# Gera matriz de confusão
disp = ConfusionMatrixDisplay.from_predictions(
        labels,
        predics,
        display_labels=range(10),
        normalize="true"
)

disp.ax_.set_title("Matriz Confusão - Dígitos MNIST")
disp.ax_.set_xlabel("Previsão")
disp.ax_.set_ylabel("Dado Real")
plt.tight_layout()
plt.savefig("confusionMatrix.png", dpi=150)
plt.show()

# Salva o modelo
torch.save(model.state_dict(), "mnist_cnn.pth")
