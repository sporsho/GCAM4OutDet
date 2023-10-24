import importlib
modules = importlib.import_module("3D_OutDet.modules")

class OutDetWithGCAM(modules.OutDet):
    def forward_to_last_activation(self, points, dist, indices):
        out = points
        for i in range(self.depth):
            if i == 0:
                out = self.convs[i](out, indices, dist=dist)
            else:
                out = self.convs[i](out, indices, dist=dist)
            if self.pool and i != self.depth - 1:
                out = self.pools[i](out, indices)
        return out


if __name__ == "__main__":
    model = OutDetWithGCAM()