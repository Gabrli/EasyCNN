import matplotlib.pyplot as plt
class TrainingVisualizer:
    
    def plot_training(self, history):
        """
        Function to dispaly in graphic format model training process.
        Docs: https://github.com/Gabrli/EasyCNN---docs
        
        :param history: value of training process history

        """
        plt.figure(figsize=(12, 5))
        def draw(parameter, index):
           plt.subplot(1,2, index)
           plt.plot(history.history[parameter], label=f'Train {parameter}')
           plt.plot(history.history[f'val_{parameter}'], label=f'Val {parameter}')
           plt.title(f'{parameter} over epochs')
           plt.legend() 

        draw('loss', 1)
        draw('accuracy', 2)
        plt.show()
    