#include <cmath>

float pad_mask_check_pixel(int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return false;
    return true;
}

void get_pad_mask(int channels,  int height,  int width,
     int ksize,  int stride, int pad, int64_t* pad_mask)
{
    int c,h,w;
    int height_col = (height - ksize) / stride + 1;
    int width_col = (width - ksize) / stride + 1;
    int filter_size = ksize*ksize*channels;
    int bit;
    int block;
    // pad just indicates that you want your windows to fit in nicely, add however many 0s as is needed (ksize/2) to make that happen,
    // means pad should either be 1 or 0 in cfg file
    if (pad){
        height_col = 1 + (height-1) / stride;
        width_col = 1 + (width-1) / stride;
        pad = ksize/2;
    }
    int output_size = height_col * width_col;
    for (c = 0; c < output_size; ++c) {
        int block_start = c * ((filter_size - 1)/64 + 1);
        int w_offset = (c*stride) % width_col;
        int h_offset = ((c*stride) / width_col) % height_col;
        for (h = 0; h < channels; ++h) {
            for (w = 0; w < (ksize*ksize); ++w) {
                int im_row = h_offset + (w / ksize);
                int im_col = w_offset + (w % ksize);
                int col_offset = (h * ksize*ksize) + w;
                // note that data col is an array of uint64 values, find which uint64 has the bit we want to set
                block = block_start + (col_offset/64);
                // now find the bit in that block that needs to be set
                bit = col_offset % 64;
                // finally, set or clear that bit
                if (pad_mask_check_pixel(height, width, channels, im_row, im_col, h, pad)) {
                    pad_mask[block] |= ((uint64_t) 1 << bit);
                } else {
                    pad_mask[block] &= ~((uint64_t) 1 << bit);
                }
            }
        }
    }
}
