#ifndef GFAZ_CLI_COMMANDS_HPP
#define GFAZ_CLI_COMMANDS_HPP

namespace gfaz::cli {

int do_compress(int argc, char *argv[]);
int do_decompress(int argc, char *argv[]);
int do_extract_path(int argc, char *argv[]);
int do_extract_walk(int argc, char *argv[]);
int do_add_haplotypes(int argc, char *argv[]);
int do_growth(int argc, char *argv[]);

} // namespace gfaz::cli

#endif
