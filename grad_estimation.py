import matplotlib.pyplot as plt
import torch
import numpy as np
import os 

from torch import autograd

def get_likelihood(model, image):

  log_p, logdet, _ = model(image)
  likelihood = log_p+logdet
  return likelihood

# def get_grad_likelihood(model, test_image):
#   log_p, logdet, _ = model(image)
#   print("Not Implemented")
#   likelihood = 0
#   return 0

def save_likelihood(path_likelihood, i, model, test_image_temoin, test_image_critic, nb_step = 0):
    bar_temoin = []
    grad_temoin = []
    bar_critic = []
    grad_critic = []
    for k in range(len(test_image_temoin)):
      model.zero_grad()
      model.requires_grad_
      current_temoin = test_image_temoin[k].unsqueeze(0)
      current_temoin = current_temoin.requires_grad_(requires_grad = True)

      likelihood = get_likelihood(model, current_temoin)
      grad = autograd.grad(likelihood, current_temoin)[0]
      grad_norm =  torch.sum(grad**2)
      bar_temoin.append(likelihood.detach().cpu().numpy())
      grad_temoin.append(grad_norm.detach().cpu().numpy())

    for k in range(len(test_image_critic)):
      model.zero_grad()
      current_critic = test_image_critic[k].unsqueeze(0)
      current_critic = current_critic.requires_grad_(requires_grad = True)

      likelihood = get_likelihood(model, current_critic)
      grad = autograd.grad(likelihood,current_critic)[0]
      grad_norm =  torch.sum(grad**2)
      bar_critic.append(likelihood.detach().cpu().numpy())
      grad_critic.append(grad_norm.detach().cpu().numpy())


    bar_critic = np.array(bar_critic).reshape(-1)
    grad_critic = np.array(grad_critic).reshape(-1)
    bar_temoin = np.array(bar_temoin).reshape(-1)
    grad_temoin = np.array(grad_temoin).reshape(-1)


    figure = plt.figure(0)
    plt.hist(bar_temoin, color = "b", alpha = 0.5)
    plt.hist(bar_critic, color = "r", alpha = 0.5)
    plt.savefig(os.path.join(path_likelihood, f"likelihood_{i}"))
    plt.close(figure)

    figure = plt.figure(0)
    plt.box_plot([bar_temoin, bar_critic], labels = ["CIFAR", "SVHN"])
    plt.savefig(os.path.join(path_likelihood, f"box_plot_likelihood_{i}"))
    plt.close(figure)
    
    figure = plt.figure(1)
    plt.hist(grad_temoin, color = "b", alpha = 0.5)
    plt.hist(grad_critic, color = "r", alpha = 0.5)
    plt.savefig(os.path.join(path_likelihood, f"grad_{i}"))
    plt.close(figure)

    
    figure = plt.figure(0)
    plt.box_plot([grad_temoin, grad_critic], labels = ["CIFAR", "SVHN"])
    plt.savefig(os.path.join(path_likelihood, f"box_plot_grad_{i}"))
    plt.close(figure)