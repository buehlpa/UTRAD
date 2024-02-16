from configurations.options import TrainOptions

def main():
    args = TrainOptions().parse()
    
    for k, v in sorted(vars(args).items()):
        print('%s: %s ' % (str(k), str(v)))
        
if __name__ == '__main__':
    main()