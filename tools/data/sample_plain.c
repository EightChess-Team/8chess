#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define INBUF (1 << 20)
#define OUTBUF (1 << 20)
#define LINEBUF 8192

static char inbuf[INBUF];
static char outbuf[OUTBUF];
static char linebuf[LINEBUF];
static char rem[LINEBUF];
static size_t outlen = 0;

static long stride, maxblocks, n = 0, k = 0;
static int keep = 0;

static void flush_out(void) {
    size_t off = 0;
    while (off < outlen) {
        ssize_t w = write(1, outbuf + off, outlen - off);
        if (w <= 0) _exit(0);
        off += (size_t)w;
    }
    outlen = 0;
}

static void process_line(const char *p, size_t len) {
    if (len >= 4 && memcmp(p, "fen ", 4) == 0) {
        n++;
        keep = (n % stride == 0);
    }
    if (!keep) return;
    if (outlen + len > OUTBUF) flush_out();
    memcpy(outbuf + outlen, p, len);
    outlen += len;
    if (p[0] == 'e' && (len == 1 || p[1] == '\n')) {
        keep = 0;
        k++;
        flush_out();
        if (maxblocks && k >= maxblocks) _exit(0);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s STRIDE [MAX_BLOCKS]\n", argv[0]);
        return 1;
    }
    stride = atol(argv[1]);
    maxblocks = argc > 2 ? atol(argv[2]) : 0;
    size_t remlen = 0;

    for (;;) {
        ssize_t r = read(0, inbuf, INBUF);
        if (r <= 0) break;
        size_t pos = 0;
        while (pos < (size_t)r) {
            char *nl = memchr(inbuf + pos, '\n', (size_t)r - pos);
            size_t seglen = nl ? (size_t)(nl - (inbuf + pos)) + 1 : (size_t)r - pos;
            if (remlen) {
                if (remlen + seglen > LINEBUF) {
                    remlen = 0;
                } else {
                    memcpy(rem + remlen, inbuf + pos, seglen);
                    remlen += seglen;
                    if (nl) {
                        process_line(rem, remlen);
                        remlen = 0;
                    }
                }
            } else if (nl) {
                process_line(inbuf + pos, seglen);
            } else {
                memcpy(rem, inbuf + pos, seglen);
                remlen = seglen;
            }
            pos += seglen;
        }
    }
    if (remlen) process_line(rem, remlen);
    flush_out();
    return 0;
}
