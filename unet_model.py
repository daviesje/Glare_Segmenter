from keras.models import Model
from keras.layers import Conv2D,Input,MaxPooling2D,Dropout,concatenate,Conv2DTranspose,Activation,BatchNormalization

def conv_block(input_tensor,n_filters,kernel_size=3,batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)    
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def down_layer(input_tensor,n_filters,kernel_size=3,batchnorm=True):
    conv = conv_block(input_tensor,n_filters,kernel_size,batchnorm)
    pool = MaxPooling2D((2,2))(conv)
    drop = Dropout(0.3)(pool)
    
    return conv,drop

def up_layer(input_tensor,concat_tensor,n_filters,kernel_size=3,batchnorm=True):
    decv = Conv2DTranspose(n_filters, (kernel_size, kernel_size), strides=(2, 2), padding='same')(input_tensor)
    conc = concatenate([decv,concat_tensor])
    drop = Dropout(0.3)(conc)
    conv = conv_block(drop,n_filters,kernel_size,batchnorm)
    
    return conv

def unet(pretrained_weights=None,input_size=(256,256,3),n_output=1,n_base=16):
    #input layer
    inputs = Input(input_size)
    
    #downward length, saving conv layers for concat
    c1,d1 = down_layer(inputs,n_base)
    c2,d2 = down_layer(d1,n_base*2)
    c3,d3 = down_layer(d2,n_base*4)
    c4,d4 = down_layer(d3,n_base*8)

    #convolution at bottom
    c5 = conv_block(d4,n_base*16)

    #upward length
    u4 = up_layer(c5,c4,n_base*8)
    u3 = up_layer(u4,c3,n_base*4)
    u2 = up_layer(u3,c2,n_base*2)
    u1 = up_layer(u2,c1,n_base)
    
    #output layer
    outputs = Conv2D(n_output,(1,1),activation='sigmoid')(u1)
    
    model = Model(inputs=[inputs],output=[outputs])

    #model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model