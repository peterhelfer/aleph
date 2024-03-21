#!/usr/bin/env python
#

class LabelSqueezer:
    def __init__(self, lbls):
        self.dict = {}
        self.revdict = {}

        next_sqz_lbl = 0
        for lbl in lbls:
            self.dict[lbl] = next_sqz_lbl
            self.revdict[next_sqz_lbl] = lbl
            next_sqz_lbl += 1

    def squeeze(self, lbl):
        """Get sqz_lbl for lbl"""
        return self.dict[lbl]

    def unsqueeze(self, sqz_lbl):
        """Get lbl for sqz_lbl"""
        return self.revdict[sqz_lbl]
        #return [k for k, v in self.dict.items() if v == sqz_lbl][0]

label_squeezer =  LabelSqueezer((1, 2, 77, 4, 77, 4))    

print(label_squeezer.squeeze(4))
print(label_squeezer.unsqueeze(3))
