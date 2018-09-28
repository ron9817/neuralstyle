from flask import Flask,render_template, request

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template("index_neural.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict_f():
    form=request.form
    if "style1" in form:
        res="style1"

    if "style2" in form:
        res="style2"

    if "style3" in form:
        res="style3"

    if "style4" in form:
        res="style4"

    f = request.files['photo']
    f.save('./static/img000.jpg')
    
    v=neural_style_transfer('./static/img000.jpg', './static/'+res+'.jpg')
    result=v.predict()
    
    
    return render_template("result.html",img='./static/img000.jpg', style='./static/'+res+'.jpg', result=result)


from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.applications import vgg19
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time
#from matplotlib import pyplot as plt

class neural_style_transfer():
  def __init__(self,target_image_path,style_refrence_image_path):
    self.target_image_path=target_image_path
    self.style_refrence_image_path=style_refrence_image_path
  
    
  def preprocess_image(self,image_path):
    img = load_img(image_path, target_size=(self.img_height, self.img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img
  
  def deprocess_image(self,x):
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x
  
  def content_loss(self,base, combination):
    return K.sum(K.square(combination - base))

#style loss
  def gram_matrix(self,x):
    features = K.batch_flatten(K.permute_dimensions(x, (2,0,1)))
    gram = K.dot(features, K.transpose(features))
    return gram

  def style_loss(self, style, combination):
    S = self.gram_matrix(style)
    C = self.gram_matrix(combination)
    channels = 3
    size = self.img_height * self.img_width
    return K.sum(K.square(S-C)) / (4. * (channels ** 2) * (size **2))

  #total variation loss
  def total_variation_loss(self,x):
    a = K.square(x[:, :self.img_height - 1, :self.img_width - 1, :] - x[:, 1:, :self.img_width - 1, :])
    b = K.square(x[:, :self.img_height - 1, :self.img_width - 1, :] - x[:, :self.img_height - 1, 1:, :])
    return K.sum(K.pow(a+b, 1.25))
  
  def predict(self):
    width, height= load_img(self.target_image_path).size
    self.img_height = 400
    self.img_width= int(width * self.img_height/height)
    target_image=K.constant(self.preprocess_image(self.target_image_path))
    style_refrence_image=K.constant(self.preprocess_image(self.style_refrence_image_path))
    combination_image=K.placeholder((1, self.img_height, self.img_width, 3))
    input_tensor=K.concatenate([target_image, style_refrence_image, combination_image], axis=0)
    model= vgg19.VGG19(input_tensor=input_tensor,weights='imagenet',include_top=False)
    print('Model Loaded')
    # Dict mapping layer names to activation tensors
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    # Name of layer used for content loss
    content_layer = 'block5_conv2'
    # Name of layers used for style loss
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    # Weights in the weighted average of the loss components
    total_variation_weight = 1e-4
    style_weight = 1.
    content_weight = 0.025

    # Define the loss by adding all components to a `loss` variable
    loss = K.variable(0.)
    layer_features = outputs_dict[content_layer]
    target_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += content_weight * self.content_loss(target_image_features,
                                          combination_features)
    for layer_name in style_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = self.style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layers)) * sl
    loss += total_variation_weight * self.total_variation_loss(combination_image)

    # Get the gradients of the generated image wrt the loss
    grads = K.gradients(loss, combination_image)[0]

    # Function to fetch the values of the current loss and the current gradients
    self.fetch_loss_and_grads = K.function([combination_image], [loss, grads])
    evaluator = Evaluator(self.img_height, self.img_width,self.fetch_loss_and_grads)

    result_prefix = './static/style_transfer_result'
    iterations = 2

    # Run scipy-based optimization (L-BFGS) over the pixels of the generated image
    # so as to minimize the neural style loss.
    # This is our initial state: the target image.
    # Note that `scipy.optimize.fmin_l_bfgs_b` can only process flat vectors.
    x = self.preprocess_image(self.target_image_path)
    x = x.flatten()
    for i in range(iterations):
        print('Start of iteration', i)
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x,
                                         fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        # Save current generated image
        img = x.copy().reshape((self.img_height, self.img_width, 3))
        img = self.deprocess_image(img)
        fname = result_prefix + '_at_iteration_%d.png' % i
        imsave(fname, img)
        end_time = time.time()
        print('Image saved as', fname)
        print('Iteration %d completed in %ds' % (i, end_time - start_time))
        
    return fname
        
        
        
        
##_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-## 
## remove when flask

#    plt.imshow(load_img(self.target_image_path, target_size=(self.img_height, self.img_width)))
#    plt.figure()

#    # Style image
#    plt.imshow(load_img('01.jpg', target_size=(self.img_height, self.img_width)))
#    plt.figure()
#
#    # Generate image
#    plt.imshow(img)
#    plt.show()


##_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-## 




class Evaluator(object):

    def __init__(self,img_height,img_width,fetch_loss_and_grads):
        self.loss_value = None
        self.grads_values = None
        self.img_height=img_height
        self.img_width=img_width
        self.fetch_loss_and_grads=fetch_loss_and_grads
                 

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, self.img_height, self.img_width, 3))
        outs = self.fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

if __name__ == "__main__":
    app.run()
