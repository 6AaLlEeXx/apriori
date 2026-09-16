from estimator.caching import resolve_representor, ShardedCache, IndexedCache, FeatureExtractor, ShardedFeatureFetch
from kernel.data import PairRecord
from pathlib import Path
import numpy as np

def sign_int():
    test_bits = ['sign_int8', 'sign_int16', 'sign_int32']
    INT = {'sign_int8': np.int8, 'sign_int16': np.int16, 'sign_int32': np.int32}

    for _ in range(10):
        shape = (int(np.random.random()*5)+1, int(np.random.random()*1000)+1)
        A = (np.random.choice(4, shape)-2).astype(np.float32)
        A[A==-2] = np.nan

        sign_int = resolve_representor('sign_int8', dim=shape[1])
        fw = sign_int.forward(A)
        assert fw.dtype == INT['sign_int8']
        res = sign_int.inverse(fw).astype(np.float32)
        assert ((res==A) + (np.isnan(A) * np.isnan(res))).all()

        sign_int = resolve_representor('sign_int16', dim=shape[1])
        fw = sign_int.forward(A)
        assert fw.dtype == INT['sign_int16']
        res = sign_int.inverse(fw).astype(np.float32)
        assert ((res==A) + (np.isnan(A) * np.isnan(res))).all()

        sign_int = resolve_representor('sign_int32', dim=shape[1])
        fw = sign_int.forward(A)
        assert fw.dtype == INT['sign_int32']
        res = sign_int.inverse(fw).astype(np.float32)
        assert ((res==A) + (np.isnan(A) * np.isnan(res))).all()

def sign_transform():
    for _ in range(20):
        shape = (int(np.random.random()*5)+1, int(np.random.random()*1000)+1)
        A = (np.random.random(shape)-0.5)*20
        
        sign = resolve_representor("sign", dim=shape[1])

        fw = sign.forward(A)
        res = sign.inverse(fw)

        assert (fw==A).all() and ((A<0.)==(res==-1)).all() and ((A>0.)==(res==1)).all() and ((A==0.)==(res==0)).all()

def thesholded_sign_transform():
    for _ in range(20):
        shape = (int(np.random.random()*5)+1, int(np.random.random()*1000)+1)
        threshold = np.random.random()*10
        A = (np.random.random(shape)-0.5)*20
        
        thresholded_sign = resolve_representor("thresholded_sign", threshold = threshold, dim=shape[1])

        fw = thresholded_sign.forward(A)
        res = thresholded_sign.inverse(fw)

        assert (fw==A).all() and ((A<-threshold)==(res==-1)).all() and ((A>threshold)==(res==1)).all() and (((A<=threshold)*(A>=-threshold))==(res==0)).all()

def multitransform():
    for _ in range(20):
        shape = (int(np.random.random()*5)+1, int(np.random.random()*1000)+1)
        threshold = np.random.random()*10
        A = (np.random.random(shape)-0.5)*20

        
        thresholded_sign = resolve_representor("multitransform", transform_name='thresholded_sign', threshold = threshold, dim=shape[1])

        fw = thresholded_sign.forward(A)
        res = thresholded_sign.inverse(fw)

        assert (fw==A).all() and ((A<-threshold)==(res==-1)).all() and ((A>threshold)==(res==1)).all() and (((A<=threshold)*(A>=-threshold))==(res==0)).all()

        sign = resolve_representor("multitransform", transform_name='sign', threshold = threshold, dim=shape[1])
        
        fw = sign.forward(A)
        res = sign.inverse(fw)

        assert (fw==A).all() and ((A<0)==(res==-1)).all() and ((A>0)==(res==1)).all() and (((A==0))==(res==0)).all()

        identity = resolve_representor("multitransform", threshold = threshold, dim=shape[1])
                
        fw = identity.forward(A)
        res = identity.inverse(fw)

        assert (fw==A).all() and (fw==res).all() and (A==res).all()


def sharded_cache():
    root_dir = Path("estimator/sharded_cache_data")

    dim = int(np.random.random()*100)+1
    cache = ShardedCache(root_dir=root_dir, dim=dim, size=5, max_blocks=2, cyclic=True, append_only=True)
    A = (np.random.random((10, dim))-0.5)*20
    cache.write([str(i) for i in range(10)], A)
    cache = ShardedCache.load(root_dir/'metadata.json')
    cache.delete("1")

    for x, a in zip(cache.retrieve(["1", "2", "3"]), [None, A[2], A[3]]):
        assert (x is None and a is None) or (a==x).all()
    cache.clean()

    dim = int(np.random.random()*100)+1
    cache = ShardedCache(root_dir=root_dir, dim=dim, size=3, max_blocks=3, cyclic=True, append_only=True)
    A = (np.random.random((10, dim))-0.5)*20
    cache.write([str(i) for i in range(10)], A)
    cache = ShardedCache.load(root_dir/'metadata.json')

    for x, a in zip(cache.retrieve(["0", "9", "1"]), [None, A[9], A[1]]):
        assert (x is None and a is None) or (a==x).all()
    cache.clean()

    dim = int(np.random.random()*100)+1
    cache = ShardedCache(root_dir=root_dir, dim=dim, size=3, max_blocks=3, cyclic=True, append_only=True)
    A = (np.random.random((10, dim))-0.5)*20
    cache.write(["0", "1", "1", "1", "2", "2", "3", "4", "5", "6"], A)
    B = (np.random.random((2, dim))-0.5)*20
    cache.write(["6", "6"], B)
    cache = ShardedCache.load(root_dir/'metadata.json')

    for x, a in zip(cache.retrieve(["0", "1", "2", "6"]), [None, A[3], A[5], B[1]]):
        assert (x is None and a is None) or (a==x).all()
    cache.clean()

    dim = int(np.random.random()*100)+1
    cache = ShardedCache(root_dir=root_dir, dim=dim, size=3, max_blocks=3, cyclic=True, append_only=False)
    A = (np.random.random((13, dim))-0.5)*20
    cache.write(["0", "1", "1", "1", "2", "2", "3", "4", "5", "6", "7", "8", "9"], A)
    B = (np.random.random((7, dim))-0.5)*20
    cache.write(["9", "9", "9", "9", "9", "9", "9"], B)
    cache = ShardedCache.load(root_dir/'metadata.json')

    for x, a in zip(cache.retrieve(["0", "1", "2", "9"]), [None, A[3], A[5], B[6]]):
        assert (x is None and a is None) or (a==x).all()
    cache.clean()

    dim = int(np.random.random()*100)+1
    cache = ShardedCache(root_dir=root_dir, dim=dim, size=3, max_blocks=3, cyclic=False, append_only=False)
    A = (np.random.random((3, dim))-0.5)*20
    cache.write(["0", "1", "2"], A)
    B = (np.random.random((7, dim))-0.5)*20
    cache.write(["9", "9", "9", "9", "9", "9", "9"], B)
    C = (np.random.random((1, dim))-0.5)*20
    cache.write(["1"], C)
    cache = ShardedCache.load(root_dir/'metadata.json')

    for x, a in zip(cache.retrieve(["0", "1", "2", "9"]), [A[0], C[0], A[2], B[6]]):
        assert (x is None and a is None) or (a==x).all()
    cache.clean()

def indexed_cache():
    root_dir = Path("estimator/indexed_cache_data")

    dim = int(np.random.random()*100)+1
    cache = IndexedCache(root_dir=root_dir, dim=dim, num_features=10)
    A = (np.random.random((10, dim))-0.5)*20
    cache.write(list(range(10)), A)
    cache = IndexedCache.load(root_dir/'metadata.json')
    cache.delete(9)
    B = (np.random.random((1, dim))-0.5)*20
    cache.write([3], B)

    for x, a in zip(cache.retrieve([9, 1, 2, 3]), [None, A[1], A[2], B[0]]):
        assert (x is None and a is None) or (a==x).all()
    cache.clean()

sharded_cache()
indexed_cache()
multitransform()
sign_int()
thesholded_sign_transform()
sign_transform()

def fetching():
    records = [
        PairRecord('001', 'train', "Tell me a joke", "Why did a chiken cross the road?"),
        PairRecord('002', 'train', "Tell me a joke", "What do you call a fake noodle? An impasta!"),
        PairRecord('003', 'train', "Tell me a fun fact", "Octopuses have three hearts."),
        PairRecord('004', 'train', "What is the capital of Japan?", "The capital of Japan is Tokyo."),
        PairRecord('005', 'train', "Give me a quick recipe for pancakes", "Mix 1 cup flour, 2 tbsp sugar, 1 tsp baking powder, 1 cup milk, and 1 egg. Cook on a hot griddle until bubbly, then flip."),
        PairRecord('006', 'val',   "Translate 'hello' to Spanish", "Hola"),
        PairRecord('007', 'train', "Write a haiku about summer", "Waves crash on hot sand / Ice cream melts in the bright sun / Cicadas buzz loud."),
        PairRecord('008', 'train', "What is the square root of 144?", "The square root of 144 is 12."),
        PairRecord('009', 'val',   "Tell me a riddle", "What has keys but no locks? A piano."),
        PairRecord('010', 'test',  "Summarize the plot of Romeo and Juliet", "Two star-crossed lovers from feuding families secretly marry, but a series of misunderstandings leads them to take their own lives, ultimately uniting their families."),
        PairRecord('011', 'train', "Give me three tips for public speaking", "1. Practice out loud. 2. Make eye contact with the audience. 3. Pause between key points."),
        PairRecord('012', 'val',   "What is the boiling point of water in Celsius?", "100°C at standard atmospheric pressure."),
        PairRecord('013', 'train', "Tell me a motivational quote", "The only way to do great work is to love what you do. – Steve Jobs"),
        PairRecord('014', 'test',  "Explain what a neural network is in one sentence", "A neural network is a computational system inspired by the brain that learns patterns from data through layers of interconnected nodes."),
        PairRecord('015', 'train', "What are the colors of the rainbow?", "Red, orange, yellow, green, blue, indigo, and violet (ROYGBIV)."),
        PairRecord('016', 'val',   "Tell me another joke", "Why don't scientists trust atoms? Because they make up everything!"),
    ]

    extractor = FeatureExtractor.smoke_extractor(adapter_path="results/adapters/20260818-233710__smollm2-1-7b-instruct__dolly__r8__s42", transform_name='thresholded_sign')
    extractor.extract_feature(PairRecord('id', 'split', "Tell me a joke", "Why did a chiken cross the road?"))
    root_dir = "estimator/sharded_cache_data"
    cache = ShardedFeatureFetch.get(   
                                    root=root_dir,
                                    input_dim=extractor.dim,
                                    size=4,
                                    max_blocks=4,
                                    append_only=True,
                                    cyclic_writes=True,
                                    representor_name='sign_int8',
                                    representor_args={}
                                    )

    cache.fetch(extractor, records)
    fetched = cache.fetch(extractor, records)
    computed = [extractor.extract_feature(r) for r in records]
    cache.clean()
    assert all([(x==y).all() for x,y in zip(fetched, computed)])

    cache = ShardedFeatureFetch.get(   
                                        root=root_dir,
                                        input_dim=extractor.dim,
                                        size=4,
                                        max_blocks=4,
                                        append_only=True,
                                        cyclic_writes=True,
                                        representor_name='multitransform',
                                        representor_args={}
                                        )
    
    cache.fetch(extractor, records)
    fetched = cache.fetch(extractor, records)
    computed = [extractor.extract_feature(r) for r in records]
    assert all([(x==y).all() for x,y in zip(fetched, computed)])

    extractor.switch_extractor_transform('identity')
    fetched = cache.fetch(extractor, records)
    computed = [extractor.extract_feature(r) for r in records]
    cache.clean()
    assert all([(x==y).all() for x,y in zip(fetched, computed)])

fetching()