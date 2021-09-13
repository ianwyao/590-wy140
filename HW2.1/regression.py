#--------------------------------
#Weihan Yao
#--------------------------------

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import random
import math
from   scipy.optimize import minimize


#------------------------
#CODE PARAMETERS
#------------------------

#USER PARAMETERS
IPLOT=True
INPUT_FILE='weight.json'
FILE_TYPE="json"


OPT_ALGO='BFGS'	#HYPER-PARAM

#UNCOMMENT FOR VARIOUS MODEL CHOICES (ONE AT A TIME)
model_type="logistic"; NFIT=4; X_KEYS=['x']; Y_KEYS=['y']
# model_type="linear";   NFIT=2; X_KEYS=['x']; Y_KEYS=['y']
# model_type="logistic"; NFIT=4; X_KEYS=['y']; Y_KEYS=['is_adult']

#RANDOM INITIAL GUESS FOR FITTING PARAMETERS
p=np.random.uniform(0.5,1.,size=NFIT)

#SAVE HISTORY FOR PLOTTING AT THE END
iteration=0; iterations=[]; loss_train=[];  loss_val=[]

#------------------------
#DATA CLASS
#------------------------

class DataClass:

    #INITIALIZE
	def __init__(self,FILE_NAME):

		if(FILE_TYPE=="json"):

			#READ FILE
			with open(FILE_NAME, errors='ignore') as f:
				self.input = json.load(f)  #read into dictionary

			#CONVERT DICTIONARY INPUT AND OUTPUT MATRICES #SIMILAR TO PANDAS DF   
			X=[]; Y=[]
			for key in self.input.keys():
				if(key in X_KEYS): X.append(self.input[key])
				if(key in Y_KEYS): Y.append(self.input[key])

			#MAKE ROWS=SAMPLE DIMENSION (TRANSPOSE)
			self.X=np.transpose(np.array(X))
			self.Y=np.transpose(np.array(Y))
			self.been_partitioned=False

			#INITIALIZE FOR LATER
			self.YPRED_T=1; self.YPRED_V=1

			#EXTRACT AGE<18
			if(model_type=="linear"):
				self.Y=self.Y[self.X[:]<18]; 
				self.X=self.X[self.X[:]<18]; 

			#TAKE MEAN AND STD DOWN COLUMNS (I.E DOWN SAMPLE DIMENSION)
			self.XMEAN=np.mean(self.X,axis=0); self.XSTD=np.std(self.X,axis=0) 
			self.YMEAN=np.mean(self.Y,axis=0); self.YSTD=np.std(self.Y,axis=0) 
		else:
			raise ValueError("REQUESTED FILE-FORMAT NOT CODED"); 

	def report(self):
		print("--------DATA REPORT--------")
		print("X shape:",self.X.shape)
		print("X means:",self.XMEAN)
		print("X stds:" ,self.XSTD)
		print("Y shape:",self.Y.shape)
		print("Y means:",self.YMEAN)
		print("Y stds:" ,self.YSTD)

	def partition(self,f_train=0.8, f_val=0.15,f_test=0.05):
		#TRAINING: 	 DATA THE OPTIMIZER "SEES"
		#VALIDATION: NOT TRAINED ON BUT MONITORED DURING TRAINING
		#TEST:		 NOT MONITORED DURING TRAINING (ONLY USED AT VERY END)


		if(f_train+f_val+f_test != 1.0):
			raise ValueError("f_train+f_val+f_test MUST EQUAL 1");

		#PARTITION DATA
		rand_indices = np.random.permutation(self.X.shape[0])
		CUT1=int(f_train*self.X.shape[0]); 
		CUT2=int((f_train+f_val)*self.X.shape[0]); 
		self.train_idx, self.val_idx, self.test_idx = rand_indices[:CUT1], rand_indices[CUT1:CUT2], rand_indices[CUT2:]
		self.been_partitioned=True

	def model(self,x,p):
		if(model_type=="linear"):   return  p[0]*x+p[1]  
		if(model_type=="logistic"): return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.0001))))

        
	def predict(self,p,meth = 'batch',t = 0):
		##Divide batch, stochastic and minibatch
		if (meth == 'batch'):
			self.YPRED_T=self.model(self.X[self.train_idx],p)
			self.YPRED_V=self.model(self.X[self.val_idx],p)
			self.YPRED_TEST=self.model(self.X[self.test_idx],p)
			
		if (meth == 'stochastic'):
			##Find the t th training data
			##t is increasing from 0 and restart again
			self.YPRED_T=self.model(self.X[self.train_idx][t],p)
			if (t >= len(self.Y[self.val_idx])):
				self.YPRED_V=self.model(self.X[self.val_idx][t % len(self.Y[self.val_idx])],p)
			else:
				self.YPRED_V=self.model(self.X[self.val_idx][t],p)
			self.YPRED_TEST=self.model(self.X[self.test_idx],p)
  
		if (meth == 'minibatch'):
			##Randomly partition the data for the first batch 
			##and then use the remainder of the training set as the second batch 
			if (t%2 == 0):
				self.minit_idx = random.sample(range(0,len(self.X[self.train_idx])),math.floor(len(self.X[self.train_idx])/2))
				self.miniv_idx = random.sample(range(0,len(self.X[self.val_idx])),math.floor(len(self.X[self.val_idx])/2))
				
				self.remaint_idx = list(set(range(len(self.Y[self.train_idx]))) - set(self.minit_idx))
				self.remainv_idx = list(set(range(len(self.Y[self.val_idx]))) - set(self.miniv_idx))
			
				self.YPRED_T=self.model(self.X[self.train_idx][self.minit_idx],p)
				self.YPRED_V=self.model(self.X[self.val_idx][self.miniv_idx],p)
				self.YPRED_TEST=self.model(self.X[self.test_idx],p)
			if (t%2 == 1):				
				self.YPRED_T=self.model(self.X[self.train_idx][self.remaint_idx],p)
				self.YPRED_V=self.model(self.X[self.val_idx][self.remainv_idx],p)
				self.YPRED_TEST=self.model(self.X[self.test_idx],p)

	def normalize(self):
		self.X=(self.X-self.XMEAN)/self.XSTD 
		self.Y=(self.Y-self.YMEAN)/self.YSTD  

	def un_normalize(self):
		self.X=self.XSTD*self.X+self.XMEAN 
		self.Y=self.YSTD*self.Y+self.YMEAN 
		self.YPRED_V=self.YSTD*self.YPRED_V+self.YMEAN 
		self.YPRED_T=self.YSTD*self.YPRED_T+self.YMEAN 
		self.YPRED_TEST=self.YSTD*self.YPRED_TEST+self.YMEAN 

	#------------------------
	#DEFINE LOSS FUNCTION
	#------------------------
	def loss(self,p,meth = 'batch',t = 0):
		global iteration,iterations,loss_train,loss_val

		#MAKE PREDICTIONS FOR GIVEN PARAM
		self.predict(p,meth,t)

		#LOSS
		if (meth == 'batch'):
			training_loss=(np.mean((self.YPRED_T-self.Y[self.train_idx])**2.0))  #MSE
			validation_loss=(np.mean((self.YPRED_V-self.Y[self.val_idx])**2.0))  #MSE
	
			loss_train.append(training_loss); loss_val.append(validation_loss)
			iterations.append(iteration)
	
			iteration+=1
			
		if (meth == 'minibatch'):
			##Calculate training loss/validation loss for every first batch and second batch 
			if (t%2 == 0):
				training_loss=(np.mean((self.YPRED_T-self.Y[self.train_idx][self.minit_idx])**2.0))  #MSE
				validation_loss=(np.mean((self.YPRED_V-self.Y[self.val_idx][self.miniv_idx])**2.0))  #MSE
		
				loss_train.append(training_loss); loss_val.append(validation_loss)
				iterations.append(iteration)
				iteration+=1
				
			if (t%2 == 1):
				training_loss=(np.mean((self.YPRED_T-self.Y[self.train_idx][self.remaint_idx])**2.0))  #MSE
				validation_loss=(np.mean((self.YPRED_V-self.Y[self.val_idx][self.remainv_idx])**2.0))  #MSE
		
				loss_train.append(training_loss); loss_val.append(validation_loss)
				iterations.append(iteration)
				iteration+=1
		
		if (meth == 'stochastic'):
			##Calculate training loss for the t th observation
			training_loss=(np.mean((self.YPRED_T-self.Y[self.train_idx][t])**2.0))  #MSE
			if (t >= len(self.Y[self.val_idx])): ##In case of index out of bounds
				validation_loss=(np.mean((self.YPRED_V-self.Y[self.val_idx][t % len(self.Y[self.val_idx])])**2.0))  #MSE
			else: 
				validation_loss=(np.mean((self.YPRED_V-self.Y[self.val_idx][t])**2.0))
			loss_train.append(training_loss)
			loss_val.append(validation_loss)
			iterations.append(iteration)
	
			iteration+=1
		return training_loss

# =============================================================================
# 	def fit(self):
# 		#TRAIN MODEL USING SCIPY MINIMIZ 
# 		res = minimize(self.loss, p, method=OPT_ALGO, tol=1e-15)
# 		popt=res.x; print("OPTIMAL PARAM:",popt)
# 
# 		#PLOT TRAINING AND VALIDATION LOSS AT END
# 		if(IPLOT):
# 			fig, ax = plt.subplots()
# 			ax.plot(iterations, loss_train, 'o', label='Training loss')
# 			ax.plot(iterations, loss_val, 'o', label='Validation loss')
# 			plt.xlabel('optimizer iterations', fontsize=18)
# 			plt.ylabel('loss', fontsize=18)
# 			plt.legend()
# 			plt.show()
# =============================================================================

	#FUNCTION PLOTS
	def plot_1(self,xla='x',yla='y'):
		if(IPLOT):
			fig, ax = plt.subplots()
			ax.plot(self.X[self.train_idx]    , self.Y[self.train_idx],'o', label='Training') 
			ax.plot(self.X[self.val_idx]      , self.Y[self.val_idx],'x', label='Validation') 
			ax.plot(self.X[self.test_idx]     , self.Y[self.test_idx],'*', label='Test') 
			ax.plot(self.X[self.train_idx]    , self.YPRED_T,'.', label='Model') 
			plt.xlabel(xla, fontsize=18);	plt.ylabel(yla, fontsize=18); 	plt.legend()
			plt.show()

	#PARITY PLOT
	def plot_2(self,xla='y_data',yla='y_predict'):
		if(IPLOT):
			fig, ax = plt.subplots()
			ax.plot(self.Y[self.train_idx]  , self.YPRED_T,'*', label='Training') 
			ax.plot(self.Y[self.val_idx]    , self.YPRED_V,'*', label='Validation') 
			ax.plot(self.Y[self.test_idx]    , self.YPRED_TEST,'*', label='Test') 
			plt.xlabel(xla, fontsize=18);	plt.ylabel(yla, fontsize=18); 	plt.legend()
			plt.show()
    
	##LOSS PLOT
	def plot_3(self,xla = 'optimizer iterations',yla = 'loss'):
		if(IPLOT):
			fig, ax = plt.subplots()
			ax.plot(iterations, loss_train,'*',label = 'Training Loss')
			ax.plot(iterations, loss_val,'*',label = 'Validation Loss')
			plt.xlabel(xla)
			plt.ylabel(yla)
			plt.legend()
			plt.show()
		
		
		
	def optimizer(self,objective, algo='GD', LR=0.01, method='batch'):
    #PARAM
		dx=0.01          					#STEP SIZE FOR FINITE DIFFERENCE
		t=0 	 							#INITIAL ITERATION COUNTER
		tmax=100000							#MAX NUMBER OF ITERATION
		tol=10**-5		#EXIT AFTER CHANGE IN F IS LESS THAN THIS 
		xi = p              #INITIAL GUESS
		print("INITAL GUESS: ",xi)
		if (algo == 'GD'):  ##Gradient Descent
		    ##The batch method need just two parameters for "objective" function
			if (method == 'batch'):
				while(t<=tmax):
		    	#NUMERICALLY COMPUTE GRADIENT 
					df_dx=np.zeros(NFIT)
					
					for i in range(0,NFIT):
						dX=np.zeros(NFIT);
						dX[i]=dx; 
						xm1=xi-dX; #print(xi,xm1,dX,dX.shape,xi.shape)
						df_dx[i]=(objective(xi,method)-objective(xm1,method))/dx
		    	    #print(xi.shape,df_dx.shape)
					xip1=xi-LR*df_dx #STEP 
		    
					if(t%10==0):
						df=np.mean(np.absolute(objective(xip1)-objective(xi)))
						print(t,"	",xi,"	","	",objective(xi)) #,df) 
		    
						if(df<tol):
							print("STOPPING CRITERION MET (STOPPING TRAINING)")
							break
		    	    #UPDATE FOR NEXT ITERATION OF LOOP
					xi=xip1  
					t=t+1
					
			##The stochastic or minibatch methods need three parameters for "objective" function		
			if (method == 'stochastic' or method == 'minibatch'):
				##Set Index
				index = 0
				while(t<=tmax):
					#NUMERICALLY COMPUTE GRADIENT 
					df_dx=np.zeros(NFIT)
					#while(index < len(self.X[self.train_idx])):
					for i in range(0,NFIT):
						dX=np.zeros(NFIT);
						dX[i]=dx; 
						xm1=xi-dX; #print(xi,xm1,dX,dX.shape,xi.shape)
						df_dx[i]=(objective(xi,method,index)-objective(xm1,method,index))/dx
		    	    #print(xi.shape,df_dx.shape)
					xip1=xi-LR*df_dx #STEP 
		    
					if(t%10==0):
						df=np.mean(np.absolute(objective(xip1)-objective(xi)))
						print(t,"	",xi,"	","	",objective(xi)) #,df) 
		    
						if(df<tol or index > tmax):
							print("STOPPING CRITERION MET (STOPPING TRAINING)")
							break
		    	    #UPDATE FOR NEXT ITERATION OF LOOP
					xi=xip1  
					index+=1
					t+=1
					##Return to 0 if index is out of the bound
					if (index >= len(self.X[self.train_idx])):
						index = 0
			##Reset these three variables since we would like to plot with all data
			self.YPRED_T=self.model(self.X[self.train_idx],xi)
			self.YPRED_V=self.model(self.X[self.val_idx],xi)
			self.YPRED_TEST=self.model(self.X[self.test_idx],xi)

        ##Gradient Descent with momentum
		if (algo == 'GD-momentum'):
			change = np.zeros(NFIT)
			##Define a momentum value
			momentum = 0.2
			if (method == 'batch'):
				while(t<=tmax):
		    	#NUMERICALLY COMPUTE GRADIENT WITH MOMENTUM
					df_dx=np.zeros(NFIT)
					
					for i in range(0,NFIT):
						dX=np.zeros(NFIT);
						dX[i]=dx; 
						xm1=xi-dX; #print(xi,xm1,dX,dX.shape,xi.shape)
						df_dx[i]=(objective(xi,method)-objective(xm1,method))/dx
		    	    #print(xi.shape,df_dx.shape)
					xip1=xi-LR*df_dx-change * momentum #STEP 
		    
					if(t%10==0):
						df=np.mean(np.absolute(objective(xip1)-objective(xi)))
						print(t,"	",xi,"	","	",objective(xi)) #,df) 
		    
						if(df<tol):
							print("STOPPING CRITERION MET (STOPPING TRAINING)")
							break
		    	    #UPDATE FOR NEXT ITERATION OF LOOP
					xi=xip1
					change = LR*df_dx + change * momentum  
					t=t+1
					
			if (method == 'minibatch' or method == 'stochastic'):
		    	#NUMERICALLY COMPUTE GRADIENT WITH MOMENTUM
				df_dx=np.zeros(NFIT)
				index = 0
				while(t <= tmax):
					for i in range(0,NFIT):
						dX=np.zeros(NFIT);
						dX[i]=dx; 
						xm1=xi-dX; #print(xi,xm1,dX,dX.shape,xi.shape)
						df_dx[i]=(objective(xi,method,index)-objective(xm1,method,index))/dx
		    	    #print(xi.shape,df_dx.shape)
					xip1=xi-LR*df_dx-change*momentum #STEP 
		    
					if(t%10==0):
						df=np.mean(np.absolute(objective(xip1)-objective(xi)))
						print(t,"	",xi,"	","	",objective(xi)) #,df) 
		    
						if(df<tol or index > tmax):
							print("STOPPING CRITERION MET (STOPPING TRAINING)")
							break
		    	    #UPDATE FOR NEXT ITERATION OF LOOP
					xi=xip1  
					index+=1
					change = LR*df_dx + change * momentum  
					t+=1
					if (index >= len(self.X[self.train_idx])):
						index = 0
			self.YPRED_T=self.model(self.X[self.train_idx],xi)
			self.YPRED_V=self.model(self.X[self.val_idx],xi)
			self.YPRED_TEST=self.model(self.X[self.test_idx],xi)        
        
        ##RMSProp algorithm
		if (algo == 'RMSProp'):
	        # list of the average square gradients for each variable
			# Set rho
			rho = 0.99
			if (method == 'batch'):
				sq_grad_avg=np.zeros(NFIT)
				while(t<=tmax):
		    	#NUMERICALLY COMPUTE MOVING AVERAGE OF THE SQUARED GRADIENT
					xip1 = np.zeros(NFIT)
					df_dx=np.zeros(NFIT)
					df_dx2=np.zeros(NFIT)
					for i in range(0,NFIT):
						dX=np.zeros(NFIT);
						dX[i]=dx; 
						xm1=xi-dX; #print(xi,xm1,dX,dX.shape,xi.shape)
						df_dx[i]=(objective(xi,method)-objective(xm1,method))/dx
		    	        #print(xi.shape,df_dx.shape)
						# squared gradient
						df_dx2[i] = df_dx[i]**2
						# update the moving average of the squared gradient
						sq_grad_avg[i] = (sq_grad_avg[i] * rho) + (df_dx2[i]* (1.0-rho)) 
					
					##Calculate new solution
					for i in range(0,NFIT):
						# calculate the step size for this variable
						alpha = dx / (1e-8 + math.sqrt(sq_grad_avg[i]))
						# calculate the new position in this variable
						value = xi[i] - alpha * df_dx[i]
						# store this variable
						xip1[i] = value
		    
					if(t%10==0):
						df=np.mean(np.absolute(objective(xip1)-objective(xi)))
						print(t,"	",xi,"	","	",objective(xi)) #,df) 
			    
						if(df<tol):
							print("STOPPING CRITERION MET (STOPPING TRAINING)")
							break
		    	    #UPDATE FOR NEXT ITERATION OF LOOP
					xi=xip1  
					t=t+1
					
			if (method == 'minibatch' or method == 'stochastic'):
		    	#NUMERICALLY COMPUTE MOVING AVERAGE OF THE SQUARED GRADIENT
				index = 0
				sq_grad_avg=np.zeros(NFIT)
				while(t <= tmax):
					df_dx=np.zeros(NFIT)
					df_dx2=np.zeros(NFIT)
					xip1=np.zeros(NFIT)
					for i in range(0,NFIT):
						dX=np.zeros(NFIT);
						dX[i]=dx; 
						xm1=xi-dX; #print(xi,xm1,dX,dX.shape,xi.shape)
						df_dx[i]=(objective(xi,method,index)-objective(xm1,method,index))/dx
						#print(xi.shape,df_dx.shape)
						# squared gradient
						df_dx2[i] = df_dx[i]**2
						# update the moving average of the squared gradient
						sq_grad_avg[i] = (sq_grad_avg[i] * rho) + (df_dx2[i]* (1.0-rho)) 
						   
					##Calculate new solution
					for i in range(0,NFIT):
						# calculate the step size for this variable
						alpha = dx / (1e-8 + math.sqrt(sq_grad_avg[i]))
						# calculate the new position in this variable
						value = xi[i] - alpha * df_dx[i]
						# store this variable
						xip1[i] = value
						
					if(t%10==0):
						df=np.mean(np.absolute(objective(xip1)-objective(xi)))
						print(t,"	",xi,"	","	",objective(xi)) #,df) 
		    
						if(df<tol or index > tmax):
							print("STOPPING CRITERION MET (STOPPING TRAINING)")
							break
		    	    #UPDATE FOR NEXT ITERATION OF LOOP
					xi=xip1  
					index+=1
					t+=1
					if (index >= len(self.X[self.train_idx])):
						index = 0
			self.YPRED_T=self.model(self.X[self.train_idx],xi)
			self.YPRED_V=self.model(self.X[self.val_idx],xi)
			self.YPRED_TEST=self.model(self.X[self.test_idx],xi)        
        
        

#------------------------
#MAIN 
#------------------------
D=DataClass(INPUT_FILE)		#INITIALIZE DATA OBJECT 
D.report()					#BASIC DATA PRESCREENING

D.partition()				#SPLIT DATA
D.normalize()				#NORMALIZE
D.optimizer(D.loss,algo='GD', LR=0.1, method='batch')
D.plot_1()					#PLOT DATA
D.plot_2()					#PLOT DATA

D.un_normalize()			#NORMALIZE
D.plot_1()					#PLOT DATA
D.plot_2()					#PLOT DATA
D.plot_3()
##### END ##########################################