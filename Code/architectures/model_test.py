import pdb 
from kerasAC.custom_losses import * 
from kerasAC.metrics import recall, specificity, fpr, fnr, precision, f1
from keras.models import load_model
import sys 
custom_objects={"recall":recall,
                "sensitivity":recall,
                "specificity":specificity,
                "fpr":fpr,
                "fnr":fnr,
                "precision":precision,
                "f1":f1,
                "ambig_binary_crossentropy":ambig_binary_crossentropy,
                "ambig_mean_absolute_error":ambig_mean_absolute_error,
                "ambig_mean_squared_error":ambig_mean_squared_error}
model_path=sys.argv[1]        
model=load_model(model_path,custom_objects=custom_objects)
print(model.summary())
pdb.set_trace()
