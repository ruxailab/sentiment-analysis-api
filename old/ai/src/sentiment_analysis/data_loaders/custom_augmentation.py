# # Import necessary libraries
# import nlpaug.augmenter.word as naw

# class CustomAugmentation(object):
#     def __init__(self, transform_type):
#         self.transform = TransformLibrary(transform_type)
        
#     def __call__(self, text):
#         return self.transform(text)
    
# class TransformLibrary(object):
#     def __init__(self, transform_type: str = "train"):
#         if transform_type == "train":
#             self.transform = naw.SynonymAug(aug_src='wordnet')  # Example using NLPAug
#         else:
#             self.transform = None  # No augmentation for other types

#     def __call__(self, text):
#         if self.transform:
#             return self.transform.augment(text)
#         return text
