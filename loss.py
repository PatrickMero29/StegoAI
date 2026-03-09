import torch
import torch.nn as nn

'''
Custom Loss Formula:
Ltotal = Ladv - λLmsg

    Ladv (Adversarial Loss): How much the image looks like static instead of a real digit.

    Lmsg (Message Loss): Bits the Extractor guessed wrong.

    λ (Lambda): Weight multiplier. Need to experiment with values. 
                If we set λ to 5.0, the AI cares 5 times more about the secret message than the image quality.
'''

class StegoLoss(nn.Module):
    def __init__(self, lambda_msg=1.0):
        # lambda_msg is a multiplier to control weight applied to the message recovery loss
        super(StegoLoss, self).__init__()
        
        # Binary Cross Entropy (BCE) Standard for checking if an image is 1.0 (Real) or 0.0 (Fake)
        self.adv_loss = nn.BCELoss()
        
        # Mean Squared Error (MSE) Measuring the distance between the original -1/1 vector and the extracted vector
        self.msg_loss = nn.MSELoss()
        
        self.lambda_msg = lambda_msg

    def forward(self, disc_judgments, real_labels, extracted_message, original_message):
        # How far off was the Discriminator 
        loss_img = self.adv_loss(disc_judgments, real_labels)
        
        # Mathematical difference between the extracted and original 
        loss_data = self.msg_loss(extracted_message, original_message)
        
        total_loss = loss_img + (self.lambda_msg * loss_data)
        
        return total_loss, loss_img, loss_data

if __name__ == "__main__":
    criterion = StegoLoss(lambda_msg=2.0) # test 2.0
    
    disc_judgments = torch.tensor([[0.8], [0.9]]) # mock data
    real_labels = torch.ones(2, 1) # target is 1.0
    
    # Extractor's guesses vs original 
    extracted_msg = torch.tensor([[0.5, -0.9], [0.8, 0.2]])
    original_msg = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
    
    total, img_l, data_l = criterion(disc_judgments, real_labels, extracted_msg, original_msg)
    
    print(f"Image Loss: {img_l.item():.4f}")
    print(f"Data Loss: {data_l.item():.4f}")
    print(f"Total Loss (Image Loss + 2.0 * Data Loss): {total.item():.4f}\n")
