# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""LastStateConnector: vLLM 0.32's ExampleHiddenStatesConnector, saving only the cache block that holds a request's
last prompt token.

The example connector saves the extracted state of every prompt token, a
file of tokens x hidden x 2 bytes per request (15 MB for 2,000 tokens of
Gemma 4 12B): each candidate of a choice question is a request whose prompt
holds the whole shared text, so a long question writes gigabytes. A choice
head reads one state, the last token's. This connector keeps, of the
request's blocks, the one that holds the last prompt token, and of its
tokens those up to it: at most one block's rows (16), the last of which is
the last token's. meepserve.vllm_engine reads that row.

vLLM imports it in its engine core's process (KVTransferConfig
kv_connector_module_path="meepserve.vllm_connector").
"""

from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
    ExampleHiddenStatesConnector)


class LastStateConnector(ExampleHiddenStatesConnector):
    def request_finished(self, request, block_ids):
        delay, params = super().request_finished(request, block_ids)
        pending = self._pending_saves.get(request.request_id)
        if pending is None or not pending.token_ids.numel():
            return delay, params
        last = pending.token_ids.shape[0] - 1
        block, offset = divmod(last, self._block_size)
        if block >= len(pending.block_ids):
            raise RuntimeError(f"request {request.request_id}: its last prompt token is in block {block}, and it holds "
                               f"{len(pending.block_ids)} blocks")
        pending.block_ids = [pending.block_ids[block]]
        pending.token_ids = pending.token_ids[last - offset:last + 1].clone()
        return delay, params
