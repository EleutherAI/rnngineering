from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
layers = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,63]
activations = []

for layer in layers:
    #plt.figure()
    activations= torch.load(f"slimpj-coordinate_activations_{layer}.pt")
    #convert list of tensors to tensor
    activations = torch.stack(activations)
    pca = PCA(n_components=2)
    activations=activations[:,0,:].cpu().numpy()
    print(activations.shape)
    projected_activations = pca.fit(activations.T)
    #plt.scatter(projected_activations[:, 0], projected_activations[:, 1])
layer=0
plt.figure()
activations= torch.load(f"slimpj-coordinate_activations_{layer}.pt")
#convert list of tensors to tensor
activations = torch.stack(activations)
pca = PCA(n_components=2)
activations=activations[:,0,:].cpu().numpy()
print(activations.shape)
projected_activations = pca.fit(activations)
#plt.scatter(projected_activations[:, 0], projected_activations[:, 1])