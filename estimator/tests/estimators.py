from estimator.estimators import Trainer, BaselineEstimator, KernelEstimator
from kernel.data import PairRecord
from pathlib import Path
import numpy as np

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

# paths = Trainer.smoke_baseline_estimator()
# estimator = BaselineEstimator(paths)

# print(estimator(records))

paths = Trainer.smoke_kernel_estimator()
estimator = KernelEstimator(paths)
print(estimator(records))