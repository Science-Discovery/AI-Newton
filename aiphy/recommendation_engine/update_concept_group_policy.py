import torch
from .concept_group import ConceptGroupPolicy


def update_network_parameters(file_path: str,
                              epoch_loss_path: str,
                              epoch: int,
                              input_vec: torch.Tensor,
                              delta_vec: torch.Tensor,
                              epoch_decay: torch.Tensor,
                              lr: float = 1e-3):
    model = ConceptGroupPolicy()
    model.load_state_dict(torch.load(file_path, weights_only=True))
    # Move the model to the device
    model.to('cuda')
    model.train()

    epoch_decay = torch.softmax(epoch_decay, dim=0)
    delta_vec = delta_vec.view(-1, 1)

    # Move the tensors to the device
    epoch_decay = epoch_decay.to('cuda')
    input_vec = input_vec.to('cuda')
    delta_vec = delta_vec.to('cuda')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(10):
        optimizer.zero_grad()
        output = model(input_vec)
        loss = torch.mul((output - delta_vec) ** 2, epoch_decay).sum()
        loss.backward()
        optimizer.step()

    model.eval()
    output = model(input_vec)
    loss = torch.mul((output - delta_vec) ** 2, epoch_decay).sum()

    # Save the updated parameters
    model.to('cpu')
    torch.save(model.state_dict(), file_path)
    print(f"Parameters updated and saved to {file_path}")

    # Save the loss
    with open(epoch_loss_path, "a") as f:
        f.write(f"epoch = {epoch}, loss = {loss.clone().cpu().detach().numpy():.2e}\n")
    print(f"Loss saved to {epoch_loss_path}")
