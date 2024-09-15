"""
Customized state encoder based on Pensieve's encoder.
"""
import torch.nn as nn


class EncoderNetwork(nn.Module):
    """
    The encoder network for encoding each piece of information of the state.
    This design of the network is from Pensieve/Genet.
    """
    def __init__(self, conv_size=4, bitrate_levels=6, embed_dim=128):
        super().__init__()
        self.past_k = conv_size
        self.bitrate_levels = 6
        self.embed_dim = embed_dim
        # self.fc1 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # last bitrate
        # self.fc2 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # current buffer size
        # self.conv3 = nn.Sequential(nn.Conv1d(1, embed_dim, conv_size), nn.LeakyReLU(), nn.Flatten())  # past k throughput
        # self.conv4 = nn.Sequential(nn.Conv1d(1, embed_dim, conv_size), nn.LeakyReLU(), nn.Flatten())  # past k download time
        # self.conv5 = nn.Sequential(nn.Conv1d(1, embed_dim, bitrate_levels), nn.LeakyReLU(), nn.Flatten())  # next chunk sizes
        # self.fc6 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # remain chunks   
         
           
        self.fc1 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  #  queue_type
        self.fc2 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  #  burst_allowance
        self.fc3 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  #  drop_probability
        self.fc4 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  #  current_queue_delay
        self.fc5 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  #  accumulated_probability
        self.fc6 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  #  average_dequeue_time
        self.fc7 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  #  total_bytes
        self.fc8 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  #  total_drops


    def forward(self, state):
        # state.shape: (batch_size, seq_len, 6, 6) -> (batch_size x seq_len, 6, 6)
        batch_size, seq_len = state.shape[0], state.shape[1]
        print("state.shape[0]",state.shape[0])
        print("state.shape[1]",state.shape[1])
        # Open the file in write mode. This will create the file if it doesn't exist or overwrite it if it does.
        with open('output_life.txt', 'w') as file:
            # Write the type of the tensor
            file.write(f"type,,state: {type(state)}\n")
            # Write the batch size and sequence length
            file.write(f"batch_size,,state.shape[0]: {state.shape[0]}\n")
            file.write(f"seq_len,,state.shape[1]: {state.shape[1]}\n")
            # Write the tensor values
            file.write("Tensor values:\n")
            file.write(f"{state}\n")

        # Optional: print to console for verification
        print("Output written to output_life.txt")
        state = state.reshape(batch_size * seq_len, 8, 1)
        
        # last_bitrate = state[..., 0:1, -1]
        # current_buffer_size = state[..., 1:2, -1]
        # throughputs = state[..., 2:3, :]
        # download_time = state[..., 3:4, :]
        # next_chunk_size = state[..., 4:5, :self.bitrate_levels]
        # remain_chunks = state[..., 5:6, -1]


        feature1 = state[..., 0:1, :]
        feature2 = state[..., 1:2, :]
        feature3 = state[..., 2:3, :]
        feature4 = state[..., 3:4, :]
        feature5 = state[..., 4:5, :]
        feature6 = state[..., 5:6, :]
        feature7 = state[..., 6:7, :]
        feature8 = state[..., 7:8, :]
        
        # features1 = self.fc1(last_bitrate).reshape(batch_size, seq_len, -1)
        # features2 = self.fc2(current_buffer_size).reshape(batch_size, seq_len, -1)
        # features3 = self.conv3(throughputs).reshape(batch_size, seq_len, -1)
        # features4 = self.conv4(download_time).reshape(batch_size, seq_len, -1)
        # features5 = self.conv5(next_chunk_size).reshape(batch_size, seq_len, -1)
        # features6 = self.fc6(remain_chunks).reshape(batch_size, seq_len, -1)

        features1 = self.fc1(feature1).reshape(batch_size, seq_len, -1)
        features2 = self.fc2(feature2).reshape(batch_size, seq_len, -1)
        features3 = self.fc3(feature3).reshape(batch_size, seq_len, -1)
        features4 = self.fc4(feature4).reshape(batch_size, seq_len, -1)
        features5 = self.fc5(feature5).reshape(batch_size, seq_len, -1)
        features6 = self.fc6(feature6).reshape(batch_size, seq_len, -1)
        features7 = self.fc7(feature7).reshape(batch_size, seq_len, -1)
        features8 = self.fc8(feature8).reshape(batch_size, seq_len, -1)



        

        return features1, features2, features3, features4, features5, features6, features7 , features8
