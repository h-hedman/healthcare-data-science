# Source from the repository root. Keep machine-specific packages out of Git.
local_library <- file.path(getwd(), ".r-library")
if (dir.exists(local_library)) .libPaths(c(local_library, .libPaths()))
if (.Platform$OS.type == "windows") {
  # Windows may inherit the unsupported POSIX locale C.UTF-8 from a shell.
  if (!l10n_info()[["UTF-8"]]) Sys.setlocale("LC_CTYPE", "English_United States.utf8")
  user_library <- file.path(Sys.getenv("LOCALAPPDATA"), "R", "win-library",
                            paste(R.version$major, strsplit(R.version$minor, ".", fixed = TRUE)[[1]][1], sep = "."))
  if (dir.exists(user_library)) .libPaths(c(local_library, user_library, .libPaths()))
}
