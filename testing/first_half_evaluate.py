from load_net import load_net
import os

this_file_path = os.path.dirname(__file__)

net, cfg = load_net(this_file_path + '/load_net/first_half.pt')