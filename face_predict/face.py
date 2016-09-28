# coding: utf8
import cPickle  
import gzip  
import os  
import sys  
import time  
  
import numpy 
  
import theano  
import theano.tensor as T  

import Image, ImageDraw


base_route = '/mnt/hgfs/kardos_test/theano/face_predict'

class LogisticRegression(object):  
    def __init__(self, input, n_in, n_out):  
  
#W大小是n_in行n_out列，b为n_out维向量。即：每个输出对应W的一列以及b的一个元素。WX+b    
#W和b都定义为theano.shared类型，这个是为了程序能在GPU上跑。  
        self.W = theano.shared(  
            value=numpy.zeros(  
                (n_in, n_out),  
                dtype=theano.config.floatX  
            ),  
            name='W',  
            borrow=True  
        )  
  
        self.b = theano.shared(  
            value=numpy.zeros(  
                (n_out,),  
                dtype=theano.config.floatX  
            ),  
            name='b',  
            borrow=True  
        )  
  
#input是(n_example,n_in)，W是（n_in,n_out）,点乘得到(n_example,n_out)，加上偏置b，  
#再作为T.nnet.softmax的输入，得到p_y_given_x  
#故p_y_given_x每一行代表每一个样本被估计为各类别的概率      
#PS：b是n_out维向量，与(n_example,n_out)矩阵相加，内部其实是先复制n_example个b，  
#然后(n_example,n_out)矩阵的每一行都加b  
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)  
  
#argmax返回最大值下标，因为本例数据集是MNIST，下标刚好就是类别。axis=1表示按行操作。  
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)  
  
#params，模型的参数       
        self.params = [self.W, self.b]  
        

#代价函数NLL  
#因为我们是MSGD，每次训练一个batch，一个batch有n_example个样本，则y大小是(n_example,),  
#y.shape[0]得出行数即样本数，将T.log(self.p_y_given_x)简记为LP，  
#则LP[T.arange(y.shape[0]),y]得到[LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,LP[n-1,y[n-1]]]  
#最后求均值mean，也就是说，minibatch的SGD，是计算出batch里所有样本的NLL的平均值，作为它的cost  
    def negative_log_likelihood(self, y):    
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])  
  
#batch的误差率  
    def errors(self, y):  
        # 首先检查y与y_pred的维度是否一样，即是否含有相等的样本数  
        if y.ndim != self.y_pred.ndim:  
            raise TypeError(  
                'y should have the same shape as self.y_pred',  
                ('y', y.type, 'y_pred', self.y_pred.type)  
            )  
        # 再检查是不是int类型，是的话计算T.neq(self.y_pred, y)的均值，作为误差率  
        #举个例子，假如self.y_pred=[3,2,3,2,3,2],而实际上y=[3,4,3,4,3,4]  
        #则T.neq(self.y_pred, y)=[0,1,0,1,0,1],1表示不等，0表示相等  
        #故T.mean(T.neq(self.y_pred, y))=T.mean([0,1,0,1,0,1])=0.5，即错误率50%  
        if y.dtype.startswith('int'):  
            return T.mean(T.neq(self.y_pred, y))  
        else:  
            raise NotImplementedError()  
    
    def predict_function(self):
        x = T.dmatrix('x')
        p_y_given_x_ = T.nnet.softmax(T.dot(x, self.W) + self.b)  
        y_pred_ = T.argmax(p_y_given_x_, axis=1)  
        return theano.function([x], y_pred_)

def load_data():
    f = open('%s/faces.pkl' % base_route)
    return cPickle.load(f)

def shared_dataset(data, borrow=True):  
    name, pixel, label = data
    shared_x = theano.shared(numpy.asarray(pixel, dtype=theano.config.floatX), borrow=borrow)  
    shared_y = theano.shared(numpy.asarray(label, dtype=theano.config.floatX), borrow=borrow)  
    return shared_x, T.cast(shared_y, 'int32')  

def sgd_optimization_mnist(learning_rate=0.05, n_epochs=1000, batch_size=100):
    data = load_data()  
    faces_x, faces_y = shared_dataset([i[600:-100] for i in data])
    valid_x, valid_y = shared_dataset([i[0:600] for i in data])

    n_faces_batches = faces_x.get_value(borrow=True).shape[0] / batch_size  
    n_valid_batches = valid_x.get_value(borrow=True).shape[0] / batch_size  

    ######################  
    # 开始建模            #  
    ######################  
    print '... building the model'  
  
  
#设置变量，index表示minibatch的下标，x表示训练样本，y是对应的label  
    index = T.lscalar()    
    x = T.matrix('x')   
    y = T.ivector('y')   
      
      
#定义分类器，用x作为input初始化。  
    classifier = LogisticRegression(input=x, n_in=20 * 20, n_out=2)  
  
  
#定义代价函数，用y来初始化，而其实还有一个隐含的参数x在classifier中。  
#这样理解才是合理的，因为cost必须由x和y得来，单单y是得不到cost的。  
    cost = classifier.negative_log_likelihood(y)  

#这里必须说明一下theano的function函数，givens是字典，其中的x、y是key，冒号后面是它们的value。  
#在function被调用时，x、y将被具体地替换为它们的value，而value里的参数index就是inputs=[index]这里给出。  
#下面举个例子：  
#比如test_model(1)，首先根据index=1具体化x为test_set_x[1 * batch_size: (1 + 1) * batch_size]，  
#具体化y为test_set_y[1 * batch_size: (1 + 1) * batch_size]。然后函数计算outputs=classifier.errors(y)，  
#这里面有参数y和隐含的x，所以就将givens里面具体化的x、y传递进去。  
    # test_model = theano.function(  
    #     inputs=[index],  
    #     outputs=classifier.errors(y),  
    #     givens={  
    #         x: test_set_x[index * batch_size: (index + 1) * batch_size],  
    #         y: test_set_y[index * batch_size: (index + 1) * batch_size]  
    #     }  
    # )  
  
  
    validate_model = theano.function(  
        inputs=[index],  
        outputs=classifier.errors(y),  
        givens={  
            x: valid_x[index * batch_size: (index + 1) * batch_size],  
            y: valid_y[index * batch_size: (index + 1) * batch_size]  
        }  
    )
# 计算各个参数的梯度  
    g_W = T.grad(cost=cost, wrt=classifier.W)  
    g_b = T.grad(cost=cost, wrt=classifier.b)  
  
#更新的规则，根据梯度下降法的更新公式  
    updates = [(classifier.W, classifier.W - learning_rate * g_W),  
               (classifier.b, classifier.b - learning_rate * g_b)]  
  
#train_model跟上面分析的test_model类似，只是这里面多了updatas，更新规则用上面定义的updates 列表。     
    train_model = theano.function(  
        inputs=[index],  
        outputs=cost,  
        updates=updates,  
        givens={  
            x: faces_x[index * batch_size: (index + 1) * batch_size],  
            y: faces_y[index * batch_size: (index + 1) * batch_size]  
        }  
    )  
  
    ###############  
    # 开始训练     #  
    ###############  
    print '... training the model'  


    patience = 5000    
    patience_increase = 2   
#提高的阈值，在验证误差减小到之前的0.995倍时，会更新best_validation_loss     
    improvement_threshold = 0.995
#这样设置validation_frequency可以保证每一次epoch都会在验证集上测试。  
    validation_frequency = min(n_faces_batches, patience / 2)  
                                  
  
    best_validation_loss = numpy.inf   #最好的验证集上的loss，最好即最小。初始化为无穷大  
    test_score = 0.  
    start_time = time.clock()  
  
    done_looping = False  
    epoch = 0  
      
#下面就是训练过程了，while循环控制的时步数epoch，一个epoch会遍历所有的batch，即所有的图片。  
#for循环是遍历一个个batch，一次一个batch地训练。for循环体里会用train_model(minibatch_index)去训练模型，  
#train_model里面的updatas会更新各个参数。  
#for循环里面会累加训练过的batch数iter，当iter是validation_frequency倍数时则会在验证集上测试，  
#如果验证集的损失this_validation_loss小于之前最佳的损失best_validation_loss，  
#则更新best_validation_loss和best_iter，同时在testset上测试。  
#如果验证集的损失this_validation_loss小于best_validation_loss*improvement_threshold时则更新patience。  
#当达到最大步数n_epoch时，或者patience<iter时，结束训练  
    while (epoch < n_epochs) and (not done_looping):  
        epoch = epoch + 1  
        for minibatch_index in xrange(n_faces_batches):  
  
            minibatch_avg_cost = train_model(minibatch_index)  
            # iteration number  
            iter = (epoch - 1) * n_faces_batches + minibatch_index  
  
            if (iter + 1) % validation_frequency == 0:  
                # compute zero-one loss on validation set  
                validation_losses = [validate_model(i)  
                                     for i in xrange(n_valid_batches)]  
                this_validation_loss = numpy.mean(validation_losses)  
  
                print(  
                    'epoch %i, minibatch %i/%i, validation error %f %%' %  
                    (  
                        epoch,  
                        minibatch_index + 1,  
                        n_faces_batches,  
                        this_validation_loss * 100.  
                    )  
                )  
  
            # if patience <= iter:  
            #     done_looping = True  
            #     break  
  
#while循环结束  
    end_time = time.clock()  
    print(  
        (  
            'Optimization complete with best validation score of %f %%,'  
            'with test performance %f %%'  
        )  
        % (best_validation_loss * 100., test_score * 100.)  
    )  
    print 'The code run for %d epochs, with %f epochs/sec' % (  
        epoch, 1. * epoch / (end_time - start_time))  
    print >> sys.stderr, ('The code for file ' +  
                          os.path.split(__file__)[1] +  
                          ' ran for %.1fs' % ((end_time - start_time)))  
    
    return classifier, [i[-100:] for i in data]

def draw(data):
    im = Image.new('RGBA', (200,200), 'white')
    draw = ImageDraw.Draw(im)
    for i in range(20):
        for j in range(20):
            poly = [(i*10, j*10), (i*10 + 10, j*10), (i*10 + 10, j*10 + 10), (i*10, j*10 + 10)]
            color = (int(data[i*20 + j] * 255),) * 3
            draw.polygon(poly, color)
    return im

if __name__ == "__main__" :

    cls, test_data = sgd_optimization_mnist()
    ppp = cls.predict_function()
    for i in range(100):
        array = numpy.array(test_data[1][i]).astype('float32')
        pred = ppp([array])[0]
        im = draw(array)
        im.save('%s/predict_result/%s_%s_%s.jpg' % (base_route, i, pred, test_data[2][i]))

    
    