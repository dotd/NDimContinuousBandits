import numpy as np
import torch


class SyntheticGaussianTorch:

    def __init__(self,
                 dim=2,
                 num_gauss=3,
                 random=np.random.RandomState(1),
                 dependance_factor=1,
                 min_x=-1.0,
                 max_x=1.0,
                 min_x_gauss=-1.0,
                 max_x_gauss=1.0,
                 std_gauss_min=0.5,
                 std_gauss_max=1.0,
                 std_factor=1):
        self.dim = dim
        self.num_gauss = num_gauss
        self.random = random
        self.dependance_factor = dependance_factor
        self.min_x = min_x
        self.max_x = max_x
        self.min_x_gauss = min_x_gauss
        self.max_x_gauss = max_x_gauss
        self.std_gauss_min = std_gauss_min
        self.std_gauss_max = std_gauss_max
        self.std_factor = std_factor
        # Centers size is num of Gauss centers x dimension
        self.centers = None
        # Variance of Gauss size is num of Gauss Centers x dimension x dimension
        self.vars = None
        self.std_, self.std_T = None, None
        self.create_random_gauss2()
        self.C_pi = (2 * np.pi) ** (-dim / 2)

    def create_centers(self):
        self.centers = torch.tensor(self.random.uniform(low=self.min_x_gauss,
                                                        high=self.max_x_gauss,
                                                        size=(self.num_gauss, self.dim)))

    def create_random_gauss(self):
        self.create_centers()
        self.std_ = torch.tensor(self.random.uniform(low=np.sqrt(self.std_gauss_min),
                                                     high=np.sqrt(self.std_gauss_max),
                                                     size=(self.num_gauss, self.dim, self.dim)))
        self.std_T = torch.transpose(self.std_, 1, 2)
        self.vars = self.dependance_factor * torch.matmul(self.std_, self.std_T) + \
                    (1 - self.dependance_factor) * torch.eye(self.dim).repeat([self.num_gauss, 1, 1]) * self.std_factor

    def create_random_gauss2(self):
        self.create_centers()
        self.vars = torch.eye(self.dim, dtype=torch.double).repeat([self.num_gauss, 1, 1])
        for g in range(self.num_gauss):
            for d in range(self.dim):
                self.vars[g, d, d] = self.random.uniform(low=self.std_gauss_min, high=self.std_gauss_max)

    def create_grid(self, num_per_dim):
        vec = torch.linspace(start=self.min_x, end=self.max_x, steps=num_per_dim)
        mesh = torch.meshgrid(*(self.dim * [vec]))
        mesh = torch.stack(mesh, 0)
        mesh = mesh.view(2, -1).T
        return mesh

    def compute_function_on_batch(self, b):
        result = 0
        for i in range(self.num_gauss):
            # print(f"i={i}")
            mu = self.centers[i]
            sigma = self.vars[i]
            sigma_m1 = torch.inverse(sigma)
            # print(mu)
            # print(sigma)
            # print(sigma_m1)
            det_norm = torch.det(sigma) ** (0.5)
            vec = (b - mu).unsqueeze(2)
            # print(f"vec=\n{vec}\nvec size={vec.shape}")
            exponent1 = torch.matmul(sigma_m1, vec).permute(0, 2, 1)
            # print(f"exponent1=\n{exponent1}\nexponent1 size={exponent1.shape}")
            exponent = torch.matmul(exponent1, vec)
            # print(f"exponent1=\n{exponent}\nexponent size={exponent.shape}")
            result += self.C_pi * det_norm * torch.exp(-exponent / 2)
        return result.squeeze()


class SyntheticGaussianNumpy:

    def __init__(self,
                 dim=2,
                 num_gauss=3,
                 random=np.random.RandomState(1),
                 dependance_factor=1,
                 min_x=-1.0,
                 max_x=1.0,
                 min_x_gauss=-1.0,
                 max_x_gauss=1.0,
                 std_gauss_min=0.5,
                 std_gauss_max=1.0,
                 std_factor=1):
        self.dim = dim
        self.num_gauss = num_gauss
        self.random = random
        self.dependance_factor = dependance_factor
        self.min_x = min_x
        self.max_x = max_x
        self.min_x_gauss = min_x_gauss
        self.max_x_gauss = max_x_gauss
        self.std_gauss_min = std_gauss_min
        self.std_gauss_max = std_gauss_max
        self.std_factor = std_factor
        # Centers size is num of Gauss centers x dimension
        self.centers = None
        # Variance of Gauss size is num of Gauss Centers x dimension x dimension
        self.vars = None
        self.vars_m1 = None
        self.det = None
        self.std_, self.std_T = None, None
        self.create_random_gauss2()
        self.C_pi = (2 * np.pi) ** (-dim / 2)

    def create_centers(self):
        self.centers = self.random.uniform(low=self.min_x_gauss,
                                           high=self.max_x_gauss,
                                           size=(self.num_gauss, self.dim))

    def create_random_gauss2(self):
        self.create_centers()
        self.vars = np.zeros(shape=(self.num_gauss, self.dim, self.dim))
        self.vars_m1 = np.zeros(shape=(self.num_gauss, self.dim, self.dim))
        self.det = np.zeros(shape=(self.num_gauss,))
        for g in range(self.num_gauss):
            for d in range(self.dim):
                self.vars[g, d, d] = self.random.uniform(low=self.std_gauss_min, high=self.std_gauss_max)
        for g in range(self.num_gauss):
            self.vars_m1[g] = np.linalg.inv(self.vars[g])
            self.det[g] = np.linalg.det(self.vars[g])

    def create_grid(self, num_per_dim):
        vec = np.linspace(start=self.min_x, stop=self.max_x, num=num_per_dim)
        mesh = np.meshgrid(*(self.dim * [vec]))
        mesh = np.stack(mesh, 0)
        mesh = mesh.reshape(2, -1).T
        return mesh

    def compute_function_on_batch(self, batch):
        result = np.zeros(shape=(batch.shape[0],))
        for idx_batch in range(batch.shape[0]):
            sample = batch[idx_batch].reshape(-1, 1)
            for idx_gaussian in range(self.num_gauss):
                mu = self.centers[idx_gaussian].reshape(-1, 1)
                sigma_m1 = self.vars_m1[idx_gaussian]
                det_norm = self.det[idx_gaussian]

                vec = (sample - mu)
                result[idx_batch] += self.C_pi * det_norm * np.exp(- np.linalg.multi_dot([vec.T, sigma_m1, vec]) / 2)
        return result.squeeze()
