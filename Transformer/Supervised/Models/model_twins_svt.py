import torch
import torch.nn as nn

# --------------------------------------------------
# Twins-SVT (simplified):
#  - LSA: Locally-grouped self-attention (non-overlapping windows)
#  - GSA: Global sub-sampled attention on downsampled tokens
#  - Hierarchical stages with patch emb and downsampling
# --------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, dim=64, patch=4, stride=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, patch, stride)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        x = self.proj(x)
        B,C,H,W = x.shape
        x = x.flatten(2).transpose(1,2)
        x = self.norm(x)
        return x, H, W

class MLP(nn.Module):
    def __init__(self, dim, ratio=4.0, drop=0.0):
        super().__init__()
        hid = int(dim*ratio)
        self.fc1 = nn.Linear(dim, hid)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid, dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x); x = self.fc2(x); x = self.drop(x); return x

class LSA(nn.Module):
    def __init__(self, dim, heads, win=7, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.win = win
        self.heads = heads
        self.scale = (dim//heads)**-0.5
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)
        self.ad = nn.Dropout(attn_drop); self.pd = nn.Dropout(proj_drop)
    def forward(self, x, H, W):
        B,N,C = x.shape
        x = x.view(B,H,W,C)
        # partition
        x = x[:, :H-(H%self.win), :W-(W%self.win), :]
        Hc = x.size(1); Wc = x.size(2)
        xw = x.view(B, Hc//self.win, self.win, Wc//self.win, self.win, C).permute(0,1,3,2,4,5).contiguous().view(-1, self.win*self.win, C)
        qkv = self.qkv(xw).reshape(xw.size(0), xw.size(1), 3, self.heads, C//self.heads).permute(2,0,3,1,4)
        q,k,v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(-1); attn = self.ad(attn)
        out = (attn @ v).transpose(1,2).reshape(xw.size(0), xw.size(1), C)
        out = self.proj(out); out = self.pd(out)
        out = out.view(B, Hc//self.win, Wc//self.win, self.win, self.win, C).permute(0,1,3,2,4,5).contiguous().view(B, Hc, Wc, C)
        # pad back if needed
        if Hc != H or Wc != W:
            padH = H - Hc; padW = W - Wc
            out = nn.functional.pad(out, (0,0,0,padW,0,padH))
        return out.view(B, H*W, C)

class GSA(nn.Module):
    def __init__(self, dim, heads, sr=4):
        super().__init__()
        self.heads=heads; self.scale=(dim//heads)**-0.5; self.sr=sr
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim*2)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x, H, W):
        B,N,C = x.shape
        q = self.q(x).reshape(B,N,self.heads,C//self.heads).permute(0,2,1,3)
        # sub-sample tokens spatially using avg-pool over 2D map
        xp = x.transpose(1,2).view(B,C,H,W)
        xs = nn.functional.avg_pool2d(xp, kernel_size=self.sr, stride=self.sr)
        Ns = xs.size(-1) * xs.size(-2)
        xs = xs.flatten(2).transpose(1,2)
        kv = self.kv(xs).reshape(B,Ns,2,self.heads,C//self.heads).permute(2,0,3,1,4)
        k,v = kv[0], kv[1]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1,2).reshape(B,N,C)
        out = self.proj(out)
        return out

class TwinsBlock(nn.Module):
    def __init__(self, dim, heads, win=7, sr=4):
        super().__init__()
        self.n1 = nn.LayerNorm(dim); self.lsa = LSA(dim, heads, win)
        self.n2 = nn.LayerNorm(dim); self.mlp = MLP(dim)
        self.n3 = nn.LayerNorm(dim); self.gsa = GSA(dim, heads, sr)
        self.n4 = nn.LayerNorm(dim); self.mlp2 = MLP(dim)
    def forward(self, x, H, W):
        x = x + self.lsa(self.n1(x), H, W)
        x = x + self.mlp(self.n2(x))
        x = x + self.gsa(self.n3(x), H, W)
        x = x + self.mlp2(self.n4(x))
        return x

class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim_in, dim_out, 3, 2, 1)
        self.norm = nn.LayerNorm(dim_out)
    def forward(self, x, H, W):
        B,N,C = x.shape
        x = x.transpose(1,2).view(B,C,H,W)
        x = self.proj(x)
        B,C,H,W = x.shape
        x = x.flatten(2).transpose(1,2)
        x = self.norm(x)
        return x, H, W

class TwinsSVT(nn.Module):
    def __init__(self, num_classes=10, embed=(64,128,256,512), depths=(1,1,3,1), heads=(2,4,8,16), win=7, sr=(4,2,1,1)):
        super().__init__()
        self.pe = PatchEmbed(3, embed[0], patch=4, stride=4)
        self.stages = nn.ModuleList()
        dim = embed[0]; H=W=None
        for i in range(4):
            blocks = nn.ModuleList([TwinsBlock(embed[i], heads[i], win=win, sr=sr[i]) for _ in range(depths[i])])
            self.stages.append(blocks)
            if i < 3:
                self.stages.append(Downsample(embed[i], embed[i+1]))
        self.norm = nn.LayerNorm(embed[-1])
        self.head = nn.Linear(embed[-1], num_classes)
    def forward(self, x):
        x,H,W = self.pe(x)
        for m in self.stages:
            if isinstance(m, nn.ModuleList):
                for b in m: x = b(x,H,W)
            else:
                x,H,W = m(x,H,W)
        x = self.norm(x)
        x = x.mean(1)
        return self.head(x)
