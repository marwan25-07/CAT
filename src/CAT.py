import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from GPyOpt.methods import BayesianOptimization
import GPy
import joblib
import pandas as pd 
from rdkit import Chem 
from mordred import Calculator, descriptors


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#transfer to avaialble device 
#prepare solvent list
solvent_df = pd.read_csv('Final_Cleaned_Minnesota Solvent Database.xls')
solvent_features = solvent_df.iloc[:,2:]


class vae(nn.Module):
    def __init__(self,weight=None, input_dim = 1621, hidden_dim = 1024, hidden_dim1 = 512, latent_dim = 256):
        super(vae, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),                                         
            nn.Linear(hidden_dim, hidden_dim1)                
        )

        self.fc_mu = nn.Linear(hidden_dim1, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim1, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # MLP property predictor (regressor head)
        self.mlp_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        if weight is not None:
            state_dict = joblib.load(weight)
            self.load_state_dict(state_dict)
            
        
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar) # standard deviation
        eps = torch.randn_like(std) # random noise
        return mu + eps * std
       
    def loss_function(self, x, reconstructed, mu, logvar, property_pred, y_true):

        # calculates the mse los sof reconstrcution loss
        reconstruction_loss = F.mse_loss(x, reconstructed)
        prop_loss = F.mse_loss(property_pred, y_true)
        

        #kl divergence approx
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = 0.5*reconstruction_loss + 2.5*prop_loss + 0.00001*kl_loss
        
        return total_loss
    
    def forward(self,x, sample_reaction):
        solvent_embedding = np.array(solvent_features.iloc[int(x)]).reshape(1,-1)     
        combined_features = np.concatenate((solvent_embedding, sample_reaction), axis=1)
        x_tensor= torch.tensor(combined_features, dtype = torch.float32)
        encoded = self.encoder(x_tensor)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparametrize(mu, logvar)
        prediction = self.mlp_head(mu)
        return prediction.detach().numpy().reshape(-1, 1)

    def optimise(self, reaction_sample):
        model = self
        model.eval()

        reactant = reaction_sample.split('>>')[0]

        #load scalar weights 
        scale = joblib.load("C:/Users/k2477436/OneDrive - King's College London/Desktop/model_details/weights/solvent_reaction_scale.pkl")
        
        #calculate descriptor values
        calc = Calculator(descriptors, ignore_3D=True)
        mol = Chem.MolFromSmiles(reactant)
        descrip = list(calc(mol))
        testing = pd.DataFrame(descrip)
        testing[0] = pd.to_numeric(testing[0], errors='coerce').fillna(0)
        sample_reaction = pd.DataFrame([testing.iloc[:, 0].tolist()]) # Flatten to one row
        
        #create solvent dictionary 
        solvent_dict = {}
        for i in range(len(solvent_df)):
            solvent_dict[solvent_df['Solvent'].iloc[i]] = solvent_df.iloc[i,2:]

        model.eval()    
        repeated_sample_reaction = np.tile(sample_reaction, (177, 1))
        optimise_samples = np.concatenate((repeated_sample_reaction, solvent_features), axis=1)
        optimise_samples_scaled = scale.transform(optimise_samples)
        optimise_samples_tensor = torch.tensor(optimise_samples_scaled, dtype=torch.float32).to(device)
        
        #predict for samples
        encoded = model.encoder(optimise_samples_tensor)
        mu = model.fc_mu(encoded)
        logvar = model.fc_logvar(encoded)
        z = model.reparametrize(mu, logvar)
        y_samples = model.mlp_head(z)
        
        #extract first 5 samples
        y_5_samples = y_samples[0:5].reshape(-1,1)

        
        model.eval()    
        repeated_sample_reaction = np.tile(sample_reaction, (177, 1))
        optimise_samples = np.concatenate((repeated_sample_reaction, solvent_features), axis=1)
        optimise_samples_tensor = torch.tensor(optimise_samples, dtype=torch.float32).to(device)
        #predict for samples
        encoded = model.encoder(optimise_samples_tensor)
        mu = model.fc_mu(encoded)
        logvar = model.fc_logvar(encoded)
        z = model.reparametrize(mu, logvar)
        y_samples = model.mlp_head(mu)
        
        #extract first 5 samples
        y_5_samples = y_samples[0:5]
        y_5_samples = y_5_samples.reshape(-1,1)

        #scale solvent data 
        solvent_dict = {}
        for i in enumerate(solvent_df):
            solvent_dict[solvent_df['Solvent'].iloc[i[0]]] = i[1]
            
        #define parameters
        kernel = GPy.kern.Matern52(input_dim=1, variance=1.0, lengthscale=1.0)
        # Define domain for optimization (example: input_dim = 20)
        domain = [{'name': 'solvent choice', 'type': 'discrete', 'domain': range(len(solvent_features))}]
        initial_x = np.arange(5).reshape(-1, 1)
        initial_y = y_5_samples.detach().cpu().numpy()
        
        optimizer = BayesianOptimization(lambda x: model.forward(x, sample_reaction),
                                        domain = domain,  
                                        model_type = 'GP',  
                                        kernel = kernel,
                                        acquisition_type = 'EI',
                                        acquisition_jitter = 0.1, 
                                        X = initial_x,  
                                        Y = initial_y,  
                                        noise_var = False, 
                                        exact_feval= False, 
                                        normalize_Y = False, 
                                        maximize = True) 
        optimizer.run_optimization(max_iter=20)
        best_index = int(optimizer.x_opt[0])
        best_solvent = solvent_df.loc[best_index]
        return best_solvent
