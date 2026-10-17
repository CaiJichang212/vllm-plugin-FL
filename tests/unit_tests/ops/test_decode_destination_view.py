"""CPU proof of the exact folded destination mapping, including offsets."""
import unittest
import torch


class DestinationViewTests(unittest.TestCase):
    def test_all_supported_batches_storage_offset_and_writes(self):
        for batch in range(1,9):
            for offset in (0,37):
                with self.subTest(batch=batch,offset=offset):
                    storage=torch.full((offset+batch*2048+19,),-13.,dtype=torch.bfloat16)
                    out=storage[offset:offset+batch*2048].view(batch,16,128)
                    old=out.view(batch,2,8,128).transpose(1,2)
                    new=out.as_strided((batch,8,2,128),(2048,128,1024,1))
                    self.assertEqual(old.shape,new.shape)
                    self.assertEqual(old.stride(),new.stride())
                    self.assertEqual(old.storage_offset(),new.storage_offset())
                    self.assertEqual(old.data_ptr(),new.data_ptr())
                    values=torch.arange(batch*2048,dtype=torch.float32).to(torch.bfloat16).reshape(batch,8,2,128)
                    new.copy_(values)
                    self.assertTrue(torch.equal(old,values))
                    self.assertTrue(torch.equal(storage[:offset],torch.full((offset,),-13.,dtype=torch.bfloat16)))
                    self.assertTrue(torch.equal(storage[offset+batch*2048:],torch.full((19,),-13.,dtype=torch.bfloat16)))


if __name__=='__main__':unittest.main()
