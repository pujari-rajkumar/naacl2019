from config import args
from utils import load_data, build_vocab, gen_submission, gen_final_submission, eval_based_on_outputs
from model import Model

if __name__ == '__main__':
    if not args.pretrained:
        print('No pretrained model specified.')
        exit(0)
    build_vocab()

    if args.dataset == 'semeval':
        dev_data = load_data('./data/dev-data-processed.json', './data/dev-data-nli.txt')
    elif args.dataset == 'multirc':
        dev_data = load_data('./data/dev_83-fixedIds-processed.json', './data/test_83-data-nli.txt')
    model_path_list = args.pretrained.split(',')
    for model_path in model_path_list:
        print('Load model from %s...' % model_path)
        args.pretrained = model_path
        model = Model(args)

        # evaluate on development dataset
        if args.dataset == 'semeval':
            dev_acc = model.evaluate(dev_data, debug=True, eval_multirc=False)
        elif args.dataset == 'multirc':
            dev_acc = model.evaluate(dev_data, debug=True, eval_multirc=True)
        print('dev accuracy: %f' % dev_acc)


