import sbt.project

ThisBuild / scalaVersion     := "2.13.14"
ThisBuild / version          := "0.1"
ThisBuild / organization     := "azeer"

val chiselVersion = "6.5.0"

val rocketVersion = "1.6-SNAPSHOT"

lazy val root = (project in file("."))
  .settings(
    name := "toyrocc",
    libraryDependencies ++= Seq(
      "org.chipsalliance" %% "chisel" % chiselVersion,
      "org.chipsalliance" %% "cde" % rocketVersion,
      "org.chipsalliance" %% s"diplomacy-$chiselVersion" % rocketVersion,
      "org.chipsalliance" %% s"hardfloat-$chiselVersion" % rocketVersion,
      "org.chipsalliance" %% "macros" % rocketVersion,
      "org.chipsalliance" %% s"rocketchip-$chiselVersion" % rocketVersion,
      "ch.epfl.scala" %% "bloop-config" % "2.0.3",
      "edu.berkeley.cs" %% "chiseltest" % "6.0.0" % Test,
      "org.scalatest" %% "scalatest" % "3.2.16" % Test
    ),
    scalacOptions ++= Seq(
      "-language:reflectiveCalls",
      "-deprecation",
      "-feature",
      "-Xcheckinit",
      "-P:chiselplugin:genBundleElements"
    ),
    addCompilerPlugin("org.chipsalliance" % "chisel-plugin" % chiselVersion cross CrossVersion.full),
	resolvers ++= Seq(
    Resolver.sonatypeRepo("snapshots"),
    Resolver.sonatypeRepo("releases"),
    Resolver.mavenLocal
  )
)
