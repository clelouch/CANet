def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

def set_arg_name(args):
    args.name += args.training_set
    if args.batch_reduce == 'multi':
        args.name += '_nei' + str(args.neighbor)
    elif args.batch_reduce == 'adapt':
        args.name += '_mag' + str(args.mag)
    if args.square:
        args.name += '_square'
    if args.second:
        args.name += '_second'
        args.name += '_ratio' + str(args.ratio)
    if args.fliprot:
        args.name += '_flip'
    if args.augmentation:
        args.name += '_rot'
    if args.stn:
        args.name += '_stn'
    if args.batch_size != 1024:
        args.name += '_bs' + str(args.batch_size)
    args.name += '_lr' + str(args.lr)
    args.name += '_epochs' + str(args.epochs)



def adjust_learning_rate(args, optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0.
        else:
            group['step'] += 1.
        group['lr'] = args.lr * (
                1.0 - float(group['step']) * float(args.batch_size) / (args.n_triplets * float(args.epochs)))
    return
