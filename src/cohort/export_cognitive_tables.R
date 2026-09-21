exports <- list(
  MMSE = list(
    file = "data/interim/adnimerge2_tables/MMSE.rda",
    cols = c(
      "RID","PTID","ORIGPROT","COLPROT",
      "VISCODE","VISCODE2","VISDATE",
      "MMSCORE","DONE","NDREASON","HAS_QC_ERROR"
    )
  ),

  CDR = list(
    file = "data/interim/adnimerge2_tables/CDR.rda",
    cols = c(
      "RID","PTID","ORIGPROT","COLPROT",
      "VISCODE","VISCODE2","VISDATE",
      "CDGLOBAL","CDRSB","HAS_QC_ERROR"
    )
  ),

  ADAS = list(
    file = "data/interim/adnimerge2_tables/ADAS.rda",
    cols = c(
      "RID","PTID","ORIGPROT","COLPROT",
      "VISCODE","VISCODE2","VISDATE",
      "TOTSCORE","TOTAL13","HAS_QC_ERROR"
    )
  ),

  NEUROBAT = list(
    file = "data/interim/adnimerge2_tables/NEUROBAT.rda",
    cols = c(
      "RID","PTID","ORIGPROT","COLPROT",
      "VISCODE","VISCODE2","VISDATE",
      "LIMMTOTAL","LDELTOTAL",
      "AVTOT1","AVTOT2","AVTOT3","AVTOT4","AVTOT5",
      "AVTOT6","AVTOTB","AVDEL30MIN","AVDELTOT",
      "HAS_QC_ERROR"
    )
  )
)

dir.create(
  "data/interim/adnimerge2_exports",
  recursive = TRUE,
  showWarnings = FALSE
)

for (name in names(exports)) {

  env <- new.env(parent = baseenv())

  load(
    exports[[name]]$file,
    envir = env
  )

  x <- get(
    name,
    envir = env
  )

  cols <- exports[[name]]$cols

  missing <- setdiff(
    cols,
    names(x)
  )

  if (length(missing) > 0) {
    stop(
      paste(
        name,
        "missing columns:",
        paste(
          missing,
          collapse = ", "
        )
      )
    )
  }

  out <- paste0(
    "data/interim/adnimerge2_exports/",
    name,
    "_cognitive_v1.csv"
  )

  write.csv(
    x[, cols],
    out,
    row.names = FALSE,
    na = ""
  )

  cat(
    name,
    ":",
    nrow(x),
    "rows exported\n"
  )
}
