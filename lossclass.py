#basic loss class
class loss:
    #calculate data and regularlization losses
    #given model output and ground truth vals
    def calculate(self, output, y):
        #calculate samples
        sample_losses=self.forward(output,y)
        #mean loss calc
        data_loss= np.mean(sample_losses)
        #return loss
        return data_loss
    #add more code in later chapters 

# cross entropy loss
class Loss_CategoricalCrossentropy(loss):
   class Loss_CategoricalCrossentropy(loss):
    #forward pass
    def forward(self, y_pred, y_true):
        # of samples in batch
        samples=len(y_pred)

        #clip data to prevent division by 0
        #clip both sides to not drag mean toward any val
        y_pred_clipped=np.clip(y_pred,1e-7, 1-1e-7)

        #prob for target val
        #only if categorical lables
        if len(y_true.shape)==1:
            correct_confidence=y_pred_clipped[
                range(samples), y_true]

        #mask values for one-hot encoded labels 
        elif len(y_true.shape)==2:
            correct_confidence=np.sum(y_pred_clipped*y_true, axis=1)
        #losses
        negative_log_likelihood= -np.log(correct_confidence)
        return negative_log_likelihood

# class inherits LOSS and performs all error calculations and can be used as an object
