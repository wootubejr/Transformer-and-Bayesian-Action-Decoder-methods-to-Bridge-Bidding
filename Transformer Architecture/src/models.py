import haiku as hk
import jax
import jax.numpy as jnp


class ActorCritic(hk.Module):
    def __init__(
        self,
        action_dim,
        activation="relu",
        model="FAIR",
    ):
        super().__init__()
        self.action_dim = action_dim
        self.activation = activation
        self.model = model

    def __call__(self, x):
        if self.activation == "relu":
            activation = jax.nn.relu
        else:
            activation = jax.nn.tanh
        if self.model == "DeepMind":
            x = hk.Linear(1024)(x)
            x = activation(x)
            x = hk.Linear(1024)(x)
            x = activation(x)
            x = hk.Linear(1024)(x)
            x = activation(x)
            x = hk.Linear(1024)(x)
            x = activation(x)
            actor_mean = hk.Linear(self.action_dim)(x)
            critic = hk.Linear(1)(x)
        elif self.model == "FAIR":
            input = x
            x = hk.Linear(200)(x)
            shortcut_1 = x
            x = activation(x)
            x = hk.Linear(200)(x)
            x = activation(x)
            x = hk.Linear(200)(x)
            x = activation(x)
            x = x + shortcut_1
            shortcut_2 = x
            x = activation(x)
            x = hk.Linear(200)(x)
            x = activation(x)
            x = hk.Linear(200)(x)
            x = activation(x)
            x = x + shortcut_2
            x = hk.Linear(200)(x)
            x = jnp.concatenate([x, input], axis=-1)
            x = hk.Linear(200)(x)
            shortcut_3 = x
            x = activation(x)
            x = hk.Linear(200)(x)
            x = activation(x)
            x = hk.Linear(200)(x)
            x = activation(x)
            x = x + shortcut_3
            shortcut_4 = x
            x = activation(x)
            x = hk.Linear(200)(x)
            x = activation(x)
            x = hk.Linear(200)(x)
            x = activation(x)
            x = x + shortcut_4
            actor_mean = hk.Linear(self.action_dim)(x)
            critic = hk.Linear(1)(x)
        return actor_mean, jnp.squeeze(critic, axis=-1)
class BridgeTransformer(hk.Module):
    def __init__(self, action_dim, d_model, num_heads, num_layers):
        super().__init__()
        self.action_dim = action_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

    def _attention_block(self, x, layer_idx):
        attn_out = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.d_model // self.num_heads,
            model_size=self.d_model,
            w_init=hk.initializers.VarianceScaling(1.0),
            name=f"attention_{layer_idx}",
        )(x, x, x)
        x = x + attn_out
        x = hk.LayerNorm(
            axis=-1,
            create_scale=True,
            create_offset=True,
            name=f"layer_norm_a_{layer_idx}"
        )(x)
        ff = hk.Linear(self.d_model * 4, name=f"ff1_{layer_idx}")(x)
        ff = jax.nn.relu(ff)
        ff = hk.Linear(self.d_model, name=f"ff2_{layer_idx}")(ff)
        x = x + ff
        x = hk.LayerNorm(
            axis=-1,
            create_scale=True,
            create_offset=True,
            name=f"layer_norm_b_{layer_idx}"
        )(x)
        return x
    
    def __call__(self, x):
        x = x.astype(jnp.float32)
        hand    = x[..., :52]
        context = x[..., 52:60]
        history = x[..., 60:]

        batch_shape = x.shape[:-1]   # (batch,) or () if unbatched
        tokens = history.reshape(*batch_shape, 35, 12)
        tokens = hk.Linear(self.d_model)(tokens)

        pos_indices    = jnp.arange(35)
        pos_embeddings = hk.Embed(35, self.d_model)(pos_indices)
        tokens         = tokens + pos_embeddings

        for i in range(self.num_layers):
            tokens = self._attention_block(tokens, layer_idx=i)

        seq_repr = jnp.mean(tokens, axis=-2)
        combined = jnp.concatenate([seq_repr, hand, context], axis=-1)
        combined = hk.Linear(self.d_model)(combined)
        combined = jax.nn.relu(combined)

        logits = hk.Linear(self.action_dim)(combined)
        value  = hk.Linear(1)(combined)

        return logits, jnp.squeeze(value, axis=-1)



def make_forward_pass(activation, model_type):
    def forward_fn(x):
        if model_type == "Transformer":
            net = BridgeTransformer(
                action_dim=38,
                d_model=256,
                num_heads=8,
                num_layers=3,
            )
        else:
            net = ActorCritic(
                38,
                activation=activation,
                model=model_type,
            )
        logits, value = net(x)
        return logits, value

    return hk.without_apply_rng(hk.transform(forward_fn))
