import numpy as np
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from data_utils import generate_alignment_data
from Neural_Networks import remove_last_layer

from utility import * 



class FedMD():
    def __init__(self, parties, public_dataset, 
                 private_data, total_private_data,  
                 private_test_data, N_alignment,
                 N_rounds, 
                 N_logits_matching_round, logits_matching_batchsize, 
                 N_private_training_round, private_training_batchsize,
                 aug = False, compress = False, select = False):
        
        self.N_parties = len(parties)
        self.public_dataset = public_dataset
        self.class_diff = np.unique(public_dataset["y"])
        self.private_data = private_data
        self.private_test_data = private_test_data
        self.N_alignment = N_alignment
        
        self.N_rounds = N_rounds
        self.aug = aug
        self.compress = compress
        self.select = select 
        self.heatmaps = np.zeros((self.N_rounds, self.N_parties, len(self.class_diff)))

        self.N_logits_matching_round = N_logits_matching_round
        self.logits_matching_batchsize = logits_matching_batchsize
        self.N_private_training_round = N_private_training_round
        self.private_training_batchsize = private_training_batchsize
        
        self.collaborative_parties = []
        self.init_result = []
        
        # print("start model initialization: ")
        for i in range(self.N_parties):
            print("model ", i)
            model_A_twin = None
            model_A_twin = clone_model(parties[i])
            model_A_twin.set_weights(parties[i].get_weights())
            model_A_twin.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-4), 
                                 loss = "sparse_categorical_crossentropy",
                                 metrics = ["accuracy"])
            
        #     print("start full stack training ... ")        
            
        #     model_A_twin.fit(private_data[i]["X"], private_data[i]["y"],
        #                      batch_size = 32, epochs = 50, shuffle=True, verbose = True,
        #                      validation_data = [private_test_data["X"], private_test_data["y"]],
        #                      callbacks=[EarlyStopping(monitor="val_accuracy", min_delta=0.0001, patience=4)]
        #                     )
            
        #     print("full stack training done")
            
            model_A = remove_last_layer(model_A_twin, loss="mean_absolute_error")
            
            self.collaborative_parties.append({"model_logits": model_A, 
                                               "model_classifier": model_A_twin,
                                               "model_weights": model_A_twin.get_weights()})
            
        #     self.init_result.append({"val_acc": model_A_twin.history.history['val_accuracy'],
        #                              "train_acc": model_A_twin.history.history['accuracy'],
        #                              "val_loss": model_A_twin.history.history['val_loss'],
        #                              "train_loss": model_A_twin.history.history['loss'],
        #                             })
            
        #     print()
        #     del model_A, model_A_twin
        # #END FOR LOOP
        
        # print("calculate the theoretical upper bounds for participants: ")
        
        # self.upper_bounds = []
        # self.pooled_train_result = []
        # for model in parties:
        #     model_ub = clone_model(model)
        #     model_ub.set_weights(model.get_weights())
        #     model_ub.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-5),
        #                      loss = "sparse_categorical_crossentropy", 
        #                      metrics = ["accuracy"])
            
        #     model_ub.fit(total_private_data["X"], total_private_data["y"],
        #                  batch_size = 32, epochs = 50, shuffle=True, verbose = True, 
        #                  validation_data = [private_test_data["X"], private_test_data["y"]],
        #                  callbacks=[EarlyStopping(monitor="val_accuracy", min_delta=0.0001, patience=4)])
            
        #     self.upper_bounds.append(model_ub.history.history["val_accuracy"][-1])
        #     self.pooled_train_result.append({"val_acc": model_ub.history.history["val_accuracy"], 
        #                                      "acc": model_ub.history.history["accuracy"]})
            
        #     del model_ub    
        # print("the upper bounds are:", self.upper_bounds)
    
    def collaborative_training(self):
        # start collaborating training    
        collaboration_performance = {i: [] for i in range(self.N_parties)}
        r = 0
        while True:
            
            if not self.aug : 
                # At beginning of each round, generate new alignment dataset
                alignment_data = generate_alignment_data(self.public_dataset["X"], 
                                                        self.public_dataset["y"],
                                                        self.N_alignment)
                
            else : 
                print("augmenting public dataset ... ")
                alpha = np.random.randint(1, 1_000_000)
                beta = np.random.randint(1, 1000)
                lambdaa = np.random.beta(alpha, alpha)
                
                np.random.seed(beta) 
                index = np.random.permutation(len(self.public_dataset["X"]))  
                new_public_dataset_x = lambdaa * self.public_dataset["X"] + (1 - lambdaa) * self.public_dataset["X"][index]
                new_public_dataset_y = self.public_dataset["y"][index]
                # new_public_dataset_y = lambdaa * self.public_dataset["y"] + (1 - lambdaa) * self.public_dataset["y"][index]

                # At beginning of each round, generate new alignment dataset
                alignment_data = generate_alignment_data(new_public_dataset_x, 
                                                        new_public_dataset_y,
                                                        self.N_alignment)
            
            

            print("round ", r)
            
            print("update logits ... ")
            # update logits
            # print("aug:{0}, compress:{1}, N_alignment:{2}".format(self.aug, self.compress, self.N_alignment))
            # print("collaborative parties", len(self.collaborative_parties))
            # print("size of alignment data {0}, length: {1}".format(size_of(alignment_data['y']), len(alignment_data["y"])))
            local_logits = []
            for d in self.collaborative_parties:
                d["model_logits"].set_weights(d["model_weights"])
                local_logits.append(d["model_logits"].predict(alignment_data["X"], verbose = 0))
            
            # print("model summary:", d['model_logits'].summary())
            # print("GT shape:", alignment_data["y"].shape)
                
            logits = aggregate(local_logits, self.compress)
            # print("size of local soft labels:{0}, size of global soft labels:{1}".format(size_of(local_logits[0]), size_of(logits)))
            # print("length of local soft labels:{0}, length of global soft labels:{1}".format(len(local_logits[0]), len(logits)))
            # print("type of local soft labels:{0}, type of global soft labels:{1}".format(local_logits[0].dtype, logits.dtype))
            # return 

            ll = np.array(local_logits) 
            gl = np.array(logits)

            print("local logits shape: ", ll.shape)       
            print("global logits shape: ", gl.shape)

            diff_l = np.power(ll - gl, 2)
            mean_diff_l = np.mean(diff_l, axis = 0)
            print("diff_l: ", diff_l.shape)
            
            client_distance_map = np.zeros((self.N_parties, len(self.class_diff))) 
            print("client_distance_map: ", client_distance_map.shape)
            print("N_parties: ", self.N_parties)
            for i in range(len(self.class_diff)):
                for c in range(self.N_parties):
                    dist = np.mean(diff_l[c, alignment_data['y'] == self.class_diff[i]])
                    client_distance_map[c, i] = dist
            
            # normalize the distance map
            # client_distance_map -= np.min(client_distance_map)
            # client_distance_map = client_distance_map / np.max(client_distance_map)
            self.heatmaps[r] = client_distance_map
            

            # test performance
            print("test performance ... ")
            
            for index, d in enumerate(self.collaborative_parties):
                y_pred = d["model_classifier"].predict(self.private_test_data["X"], verbose = 0).argmax(axis = 1)
                collaboration_performance[index].append(np.mean(self.private_test_data["y"] == y_pred))
                
                print(collaboration_performance[index][-1])
                del y_pred
                
                
            r+= 1
            if r >= self.N_rounds:
                break
                
                
            print("updates models ...")
            for index, d in enumerate(self.collaborative_parties):
                print("model {0} starting alignment with public logits... ".format(index))
                
                
                weights_to_use = None
                weights_to_use = d["model_weights"]

                d["model_logits"].set_weights(weights_to_use)
                print("fitting")
                d["model_logits"].fit(alignment_data["X"], logits, 
                                      batch_size = self.logits_matching_batchsize,  
                                      epochs = self.N_logits_matching_round, 
                                      shuffle=True, verbose = 0)
                d["model_weights"] = d["model_logits"].get_weights()
                print("model {0} done alignment".format(index))

                print("model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                print("model {0} done private training. \n".format(index))
            #END FOR LOOP
        
        #END WHILE LOOP
        return collaboration_performance


        