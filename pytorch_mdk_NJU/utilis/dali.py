from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

class SimplePipeline(Pipeline):
    def __init__(self, image_dir, batch_size, num_threads, device_id):
        super(SimplePipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input = ops.FileReader(file_root = image_dir)
        # instead of path to file directory file with pairs image_name image_label_value can be provided
        # self.input = ops.FileReader(file_root = image_dir, file_list = image_dir + '/file_list.txt')
        self.decode = ops.ImageDecoder(device = 'mixed', output_type = types.RGB)
        self.transpose = ops.Transpose(device = 'gpu', perm=[2,0,1])
    def define_graph(self):
        jpegs, _ = self.input()
        images = self.decode(jpegs)
        # images = images.gpu()
        images = self.transpose(images)
        return images

# class SimpleDataset(Dataset):
#     def __init__(self, input_path, img_size = 256):
#         super(SimpleDataset, self).__init__()
#         self.input_list = []
#         self.label_list = []
#         self.num = 0
#         self.img_size = img_size

#         for _ in range(30):
#             for i in os.listdir(input_path):
#                 input_img = input_path + i
#                 self.input_list.append(input_img)
#                 self.num = self.num + 1

#     def __len__(self):
#         return self.num

#     def __getitem__(self, idx):
#         img = np.array(Image.open(self.input_list[idx]))
#         input_np = img.astype(np.float32).transpose(2, 0, 1) / 255.0
#         input_tensor = torch.from_numpy(input_np)
#         return input_tensor

# class MyDataset(Dataset):
#     def __init__(self, input_path, img_size = 256):
#         super(MyDataset, self).__init__()
#         self.input_list = []
#         self.label_list = []
#         self.num = 0
#         self.img_size = img_size

#         for i in os.listdir(input_path):
#             input_img = input_path + i
#             self.input_list.append(input_img)
#             self.num = self.num + 1

#     def __len__(self):
#         return self.num

#     def __getitem__(self, idx):
#         img = np.array(Image.open(self.input_list[idx]))
#         x = np.random.randint(0, img.shape[0] - self.img_size)
#         y = np.random.randint(0, img.shape[1] - self.img_size)
#         input_np = img[x:x + self.img_size, y:y + self.img_size, :].astype(np.float32).transpose(2, 0, 1) / 255.0
#         input_tensor = torch.from_numpy(input_np)
#         return input_tensor

# pipe = SimplePipeline(image_dir ,batch_size, num_threads = 1, device_id = 0)
# pipe.build()

# pipe_out = pipe.run()