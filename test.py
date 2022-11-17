from wandb_para import wandb_para as wp
def test():
    global x,y
    x = wp.scattering_sweep['parameters']
    y = wp.scattering_sweep['parameters']

def new():
    global x,y
    print(x,y)


global a

def assign():
    global a
    a = 3


class new_test():
    def ttt(self):
        global a
        print(a)
print(wp.scattering_sweep['parameters']['J']['values'])
