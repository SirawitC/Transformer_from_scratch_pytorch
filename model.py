import torch
import torch.nn as nn 
import math

class InputEmbedding(nn.Module):
    """ Create an instance for input embedding component.

    This layer transform the input token to the corresponding embedding vector of size (d_model x 1).

    Attributes:
        d_model: An Integer representing the output dimension of the embedding layer.
        vocab_size: An Integer representing the size of vocabulary.
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """ Initialize the input embedding layer.

        Args:
            d_model: An Integer representing the output dimension of the embedding layer.
            vocab_size: An Integer representing the size of vocabulary.
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """ Forward function for input embedding layer.

        This function will transform the input token sequence x to the corresponding embedding vector of size (d_model x 1).

        Args:
            x: A tensor representing the input token sequence

        Returns:
            A tensor representing the embedding vector
        """
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    """ Create an instance for positional encoding component.

    This layer add positional encoding informatiobn into the embedding vector.

    Attributes:
        d_model: An Integer representing the dimension of the model.
        seq_len: An Integer representing the length of the input sequence.
        dropout: A Float representing the dropout rate. 
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """ Initialize the positional encoding layer.

        Args:
            d_model: An Integer representing the dimension of the model.
            seq_len: An Integer representing the length of the input sequence.
            dropout: A Float representing the dropout rate.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # initialize matrix of size (seq_len X d_model)
        pe = torch.zeros(seq_len, d_model)

        # initialize the vector of position of size (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        # apply sin to even and cos to odd position
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # size (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """ Forward function for positional encoding layer.

        This function will add positional encoding information into the embedding vector.

        Args:
            x: A tensor representing the embedding vector.

        Returns:
            A tensor representing the output of the positional encoding layer.
        """
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    """ Create an instance for layer normalization component.

    This layer performe the layer normalization on the input.

    Attributes:
        epsilon: A Float representing the epsilon value.
        alpha: A Float representing the alpha value (Multiplicative).
        bias: A Float representing the bias value (Additive).
    """
    def __init__(self, epsilon: float = 10**-6) -> None:
        """ Initialize the layer normalization layer.

        Args:
            epsilon: A Float representing the epsilon value. If not provided, the default value is 10**-6.
        """
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplicative
        self.bias = nn.Parameter(torch.zeros(1)) # Additive

    def forward(self, x):
        """ Forward function for layer normalization layer.

        This function will normalize the input tensor according to the statistics of the input tensor.

        Args:
            x: A tensor representing the input.

        Returns:
            A tensor representing the normalized tensor.
        """
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.epsilon) + self.bias
    
class FeedForwardBlock(nn.Module):
    """ Create an instance for feed forward block component.

    This layer pass the input through two linear layers performing the affine transformation on the input data.

    Attributes:
        d_model: An Integer representing the dimension of the model.
        d_ff: An Integer representing the dimension of the feed forward layer.
        dropout: A Float representing the dropout rate.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """ Initialize the feed forward block layer.

        Args:
            d_model: An Integer representing the dimension of the model.
            d_ff: An Integer representing the dimension of the feed forward layer.
            dropout: A Float representing the dropout rate.
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # W2 and b2

    def forward(self, x):
        """ Forward function for feed forward block layer.

        This function will pass the input through two linear layers performing the affine transformation on the input data.

        Args:
            x: A tensor representing the input.

        Returns:
            A tensor representing the output of the feed forward block layer.
        """
        x = self.linear1(x) # (batch, seq_len, d_model) --> (batch, seq_len, d_ff)
        x = torch.relu(x)  
        x = self.dropout(x)  
        out = self.linear2(x) # (batch, seq_len, d_ff) --> (batch, seq_len, d_model) 
        return out

class MultiHeadAttentionBlock(nn.Module):
    """ Create an instance for multi head attention block component.
    
    This layer perform the multi head attention on the input data.

    Attributes:
        d_model: An Integer representing the dimension of the model..
        num_head: An Integer representing the number of heads of the multi head attention.
        dropout: A Float representing the dropout rate.
    """
    def __init__(self, d_model: int, num_head: int, dropout: float) -> None:
        """ Initialize the multi head attention block layer.

        Args:
            d_model: An Integer representing the dimension of the model.
            num_head: An Integer representing the number of heads of the multi head attention.
            dropout: A Float representing the dropout rate.
        """
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        assert d_model % num_head == 0, 'd_model must be divisible by num_head'

        self.d_k = d_model // num_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """ Compute the attention weights and output given the query, key, value and mask.

        Args:
            query: A tensor representing the query. 
            key: A tensor representing the key. 
            value: A tensor representing the value. 
            mask: A tensor representing the mask. 
            dropout: A Float representing the dropout rate.

        Returns:
            out: A tensor representing the output of multi-head attention layer. 
            attention_score: A tensor representing the attention score.
        """
        d_k = query.shape[-1]

        # (batch, num_head, seq_len, d_k) --> (batch, num_head, seq_len, seq_len)
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e9)
        attention_score = attention_score.softmax(dim=-1) # (batch, num_head, seq_len, seq_len)
        if dropout is not None:
            attention_score = dropout(attention_score)

        return (attention_score @ value), attention_score
    
    def forward(self, q, k, v, mask):
        """Forward function for multi-head attention layer.

        This function will compute the attention score and output given the query, key, value and mask.

        Args:
            q: A tensor representing the query. 
            k: A tensor representing the key. 
            v: A tensor representing the value. 
            mask: A tensor representing the mask. 

        Returns:
            x: A tensor representing the output of multi-head attention layer. 
            attention_score: A tensor representing the attention score.
        """
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, num_head, d_k) --> (batch, num_head, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_head, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.num_head, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.num_head, self.d_k).transpose(1,2)

        x, self.attention_score = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (batch, num_head, seq_len, d_k) --> (batch, seq_len, num_head, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.num_head*self.d_k)

        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    """ Create an instance for residual connection component.

    This function will perform the residual connection connecting the input and output of the sublayer to the following layer.

    Attributes:
        dropout: A Float representing the dropout rate.
        norm: A LayerNormalization layer.
    """
    def __init__(self, dropout: float) -> None:
        """ Initialize the residual connection layer.

        Args:
            dropout: A Float representing the dropout rate.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        """ Forward function for residual connection layer.

        This function will perform the residual connection connecting the input and output of the sublayer to the following layer.

        Args:
            x: A tensor representing the input.
            sublayer: A callable representing the sublayer.

        Returns:
            A tensor representing the output of the residual connection layer.
        """
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    """ Create an instance for encoder block component.

    Create a encoder block with self attention and feed forward block.

    Attributes:
        self_attention_block: A MultiHeadAttentionBlock layer.
        feed_forward_block: A FeedForwardBlock layer.
        residual_connection: A ResidualConnection layer.
    """
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        """ Initialize the encoder block layer.

        Args:
            self_attention_block: A MultiHeadAttentionBlock layer.
            feed_forward_block: A FeedForwardBlock layer.
            dropout: A Float representing the dropout rate.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        """ Forward function for encoder block layer.

        This function will pass the input through multi-head attention, feed forward block and perform the residual connection.

        Args:
            x: A tensor representing the input.
            src_mask: A tensor representing the source mask.

        Returns:
            A tensor representing the output of the encoder block layer.
        """
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    """ Create an instance for encoder component.

    Create a encoder with multiple encoder block.

    Attributes:
        layers: A ModuleList of EncoderBlock layers.
        norm: A LayerNormalization layer.
    """
    def __init__(self, layers: nn.ModuleList) -> None:
        """ Initialize the encoder layer.

        Args:
            layers: A ModuleList of EncoderBlock layers.
        """
        super().__init__()
        self.layers = layers
        self.norm  = LayerNormalization()

    def forward(self, x, mask):
        """ Forward function for encoder layer.

        This function will pass the input through multiple encoder block and perform the layer normalization.

        Args:
            x: A tensor representing the input.
            mask: A tensor representing the source mask.

        Returns:
            A tensor representing the output of the encoder layer.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    """ Create an instance for decoder block component.

    Create a decoder block with self attention, cross attention, feed forward block, and the residual connection.

    Attributes:
        self_attention: A MultiHeadAttentionBlock layer.
        cross_attention: A MultiHeadAttentionBlock layer.
        feed_forward_block: A FeedForwardBlock layer.
        residual_connection: A ResidualConnection layer.
    """
    def __init__(self, self_attention: MultiHeadAttentionBlock, cross_attention: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        """ Initialize the decoder block layer.

        Args:
            self_attention: A MultiHeadAttentionBlock layer.
            cross_attention: A MultiHeadAttentionBlock layer.
            feed_forward_block: A FeedForwardBlock layer.
            dropout: A Float representing the dropout rate.
        """
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """ Forward function for decoder block layer.

        This function will pass the input through multiple attention and feed forward block, and perform the residual connection.

        Args:
            x: A tensor representing the input.
            encoder_output: A tensor representing the output of the encoder layer.
            src_mask: A tensor representing the source mask.
            tgt_mask: A tensor representing the target mask.

        Returns:
            A tensor representing the output of the decoder block layer.
        """
        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    """ Create an instance for decoder component.

    Create a decoder with multiple decoder block.

    Attributes:
        layers: A ModuleList of DecoderBlock layers.
        norm: A LayerNormalization layer.
    """
    def __init__(self, layers: nn.ModuleList) -> None:
        """ Initialize the decoder layer.

        Args:
            layers: A ModuleList of DecoderBlock layers.
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """ Forward function for decoder layer.

        This function will pass the input through multiple decoder block, and perform the layer normalization.

        Args:
            x: A tensor representing the input.
            encoder_output: A tensor representing the output of the encoder layer.
            src_mask: A tensor representing the source mask.
            tgt_mask: A tensor representing the target mask.

        Returns:
            A tensor representing the output of the decoder layer.
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    """ Create an instance for projection layer component.

    Create a projection layer with log softmax and linear layer to project the output of the decoder to the vocabulary.

    Attributes:
        projection: A Linear layer.
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """ Initialize the projection layer.

        Args:
            d_model: An Integer representing the dimension of the model.
            vocab_size: An Integer representing the size of vocabulary.
        """
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        """ Forward function for projection layer.

        This function will project the output of the decoder to the vocabulary size and apply the log softmax.

        Args:
            x: A tensor representing the output of the decoder.

        Returns:
            A tensor representing the output of the projection layer.
        """
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.projection(x), dim=-1)
    
class Transformer(nn.Module):
    """ Create an instance for transformer model.

    Create a transformer model with encoder, decoder, source embedding, target embedding, source positional encoding, target positional encoding, and projection layer.

    Attributes:
        encoder: A Encoder layer.
        decoder: A Decoder layer.
        src_embed: A InputEmbedding layer.
        tgt_embed: A InputEmbedding layer.
        src_pos: A PositionalEncoding layer.
        tgt_pos: A PositionalEncoding layer.
        projection_layer: A ProjectionLayer layer.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        """ Initialize the transformer model.

        Args:
            encoder: A Encoder layer.
            decoder: A Decoder layer.
            src_embed: A InputEmbedding layer.
            tgt_embed: A InputEmbedding layer.
            src_pos: A PositionalEncoding layer.
            tgt_pos: A PositionalEncoding layer.
            projection_layer: A ProjectionLayer layer.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        """ Encode the source sequence.

        This function will embed the source sequence, add the positional encoding, and feed it into the encoder.

        Args:
            src: A tensor representing the source sequence.
            src_mask: A tensor representing the source mask.

        Returns:
            A tensor representing the output of the encoder.
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        """ Decode the target sequence.

        This function will embed the target sequence, add the positional encoding, and feed it into the decoder.

        Args:
            encoder_output: A tensor representing the output of the encoder.
            src_mask: A tensor representing the source mask.
            tgt: A tensor representing the target sequence.
            tgt_mask: A tensor representing the target mask.

        Returns:
            A tensor representing the output of the decoder.
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        """ Project the output of the decoder to the target vocabulary.

        This function will apply the projection layer to the output of the decoder, and return the result.

        Args:
            x: A tensor representing the output of the decoder.

        Returns:
            A tensor representing the output of the projection layer.
        """
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    """Build a transformer model.

    Args:
        src_vocab_size (int): The number of unique words in the source language.
        tgt_vocab_size (int): The number of unique words in the target language.
        src_seq_len (int): The length of the sequence in the source language.
        tgt_seq_len (int): The length of the sequence in the target language.
        d_model (int, optional): The dimensionality of the model. Defaults to 512.
        N (int, optional): The number of encoder and decoder layers. Defaults to 6.
        h (int, optional): The number of heads in the multi-head attention. Defaults to 8.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        d_ff (int, optional): The dimensionality of the feed forward layer. Defaults to 2048.

    Returns:
        Transformer: The transformer model.
    """
    # Create embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # Create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create Encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create Decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create Encoder and Decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create Transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer