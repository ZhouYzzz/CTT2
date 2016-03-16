#!/usr/bin/python

from net.net_first_half import net_first_half as net

print 'Net Loaded', net

for (name, blob) in net.blobs.iteritems():
	print name, blob.data.shape