---
layout: post
title: Self-balancing Binary Search Tree
---

## AVL-Tree
Here we briefly introduce one of the self-balancing binary search tree (AVL-Tree).

### Four basic rotations
Note that the red nodes are key pattern in rotations. Whether the blue nodes exist or not doesn't matter. 
{% sidenote 1 'For the c++ implementation, please check the AVLTree project in VS2017.'%}

The basic left and right rotations:
{% maincolumn 'assets/algorithm/AVLTree_right_rotation.png'%}
{% maincolumn 'assets/algorithm/AVLTree_left_rotation.png'%}

The left-right rotation and right-left rotation:
{% maincolumn 'assets/algorithm/AVLTree_left_right_rotation.png'%}
The left-right rotation: it just first left rotate 5, 6. And then right rotate 10, 6, 5. 
Note that the 5.5 could be the right child of 6, and the whole routine is still the same.

{% maincolumn 'assets/algorithm/AVLTree_right_left_rotation.png'%}
The right-left rotation: it just first right rotate 15, 13. And then left rotate 10, 13, 15.



