from random import choice, random
from time import sleep


def get_random_key():
    '''Returns randomized key, with randomized jitter delay added
    This is to distribute load across keys and avoid throttling
    of the same key at the same time for multiple concurrent calls'''
    keys = [1,2,3,4,5]
    sleep(random())
    return choice(keys)

if __name__ == '__main__':
    print(get_random_key())
