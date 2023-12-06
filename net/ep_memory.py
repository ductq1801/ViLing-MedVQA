import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_, normal
import torch.nn.functional as F

def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()

    return lin

class ControlUnit(nn.Module):
    def __init__(self, dim, max_step):
        super().__init__()

        self.position_aware = nn.ModuleList()
        for i in range(max_step):
            self.position_aware.append(linear(dim, dim))

        self.control_question = linear(2*dim, dim)
        self.context_proj = nn.Parameter(torch.rand((30,169)))
        self.attn = linear(dim, 1)

        self.dim = dim

    def forward(self, step, context, question, control):
        position_aware = self.position_aware[step](question)

        control_question = torch.cat([control, position_aware], 2)
        control_question = self.control_question(control_question)
        p_context = torch.einsum('bnc,vn->bvc',context,self.context_proj)

        context_prod = control_question * p_context
        attn_weight = self.attn(context_prod)

        attn = F.softmax(attn_weight, 1)

        next_control = (attn * p_context)

        return next_control


class ReadUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.mem = linear(dim, dim)
        self.concat = linear(dim * 2, dim)
        self.attn = linear(dim, dim)
        self.know_proj = nn.Parameter(torch.rand((30,169)))
    def forward(self, memory, know, control):

        mem = self.mem(memory)
        know_p = torch.einsum('bnc,vn->bvc',know,self.know_proj)

        concat = self.concat(torch.cat([mem * know_p, know_p], 2))

        attn = concat * control

        attn = self.attn(attn).squeeze(2)

        attn = F.softmax(attn, 1).unsqueeze(1)

        read = (attn * know_p).sum(1)

        return read


class WriteUnit(nn.Module):
    def __init__(self, dim, self_attention=False, memory_gate=False):
        super().__init__()

        self.concat = linear(dim * 2, dim)

        if self_attention:
            self.attn = linear(dim, 1)
            self.mem = linear(dim, dim)

        if memory_gate:
            self.control = linear(dim, 1)

        self.self_attention = self_attention
        self.memory_gate = memory_gate

    def forward(self, memories, retrieved, controls):
        prev_mem = memories
        print(retrieved.shape)
        print(prev_mem.shape)
        concat = self.concat(torch.cat([retrieved, prev_mem], 2))
        next_mem = concat

        if self.self_attention:
            controls_cat = torch.stack(controls[:-1], 2)
            attn = controls[-1].unsqueeze(2) * controls_cat
            attn = self.attn(attn.permute(0, 2, 1))
            attn = F.softmax(attn, 1).permute(0, 2, 1)

            memories_cat = torch.stack(memories, 2)
            attn_mem = (attn * memories_cat).sum(2)
            next_mem = self.mem(attn_mem) + concat

        if self.memory_gate:
            control = self.control(controls[-1])
            gate = F.sigmoid(control)
            next_mem = gate * prev_mem + (1 - gate) * next_mem

        return next_mem


class MACUnit(nn.Module):
    def __init__(self, dim,mem = 30, max_step=12,
                self_attention=False, memory_gate=False,
                dropout=0.15):
        super().__init__()

        self.control = ControlUnit(dim, max_step)
        self.read = ReadUnit(dim)
        self.write = WriteUnit(dim, self_attention, memory_gate)

        self.mem_0 = nn.Parameter(torch.zeros(1,mem, dim))
        self.control_0 = nn.Parameter(torch.zeros(1,mem, dim))

        self.dim = dim
        self.max_step = max_step
        self.dropout = dropout
        self.mem = mem
    def get_mask(self, x, dropout):
        mask = torch.empty_like(x).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)

        return mask

    def forward(self, context, question, knowledge):
        b_size = question.size(0)

        control = self.control_0.expand(b_size,self.mem, self.dim)
        memory = self.mem_0.expand(b_size, self.mem,self.dim)

        if self.training:
            control_mask = self.get_mask(control, self.dropout)
            memory_mask = self.get_mask(memory, self.dropout)
            control = control * control_mask
            memory = memory * memory_mask

        controls = control
        memories = memory

        for i in range(self.max_step):
            control = self.control(i, context, question, control)
            if self.training:
                control_mask = self.get_mask(control, self.dropout)
                control = control * control_mask
              
            controls=controls + control

            read = self.read(memories, knowledge, controls)
            print(read.shape)
            memory = self.write(memories, read, controls)
            if self.training:
                memory = memory * memory_mask
            memories= memories + memory

        return memory