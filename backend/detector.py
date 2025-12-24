import torch
from model import CNN


class classifire:
    def __init__(self,model_path,class_names):
        self.device = device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model =CNN(num_classes=len(class_names)).to(self.device)
        self.model.load_state_dict(torch.load(model_path,map_location=self.device))
        self.model.eval()
        
        self.class_names =class_names
        
    def predict(self,input_tensor):
        input_tensor =input_tensor.to(self.device)

        with torch.no_grad():
            outputs =self.model(input_tensor)
            probs = torch.softmax(outputs,dim=1)
            claas_id =torch.argmax(probs,dim=1).item()
            confidence =probs[0,claas_id].item()
            
        return self.class_names[claas_id],confidence