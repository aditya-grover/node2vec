package com.navercorp.graph

import scala.collection.mutable.ArrayBuffer

object GraphOps {
  def setupAlias(nodeWeights: Array[(Long, Double)]): (Array[Int], Array[Double]) = {
    val K = nodeWeights.length
    val J = Array.fill(K)(0)
    val q = Array.fill(K)(0.0)

    val smaller = new ArrayBuffer[Int]()
    val larger = new ArrayBuffer[Int]()

    val sum = nodeWeights.map(_._2).sum
    nodeWeights.zipWithIndex.foreach { case ((nodeId, weight), i) =>
      q(i) = K * weight / sum
      if (q(i) < 1.0) {
        smaller.append(i)
      } else {
        larger.append(i)
      }
    }

    while (smaller.nonEmpty && larger.nonEmpty) {
      val small = smaller.remove(smaller.length - 1)
      val large = larger.remove(larger.length - 1)

      J(small) = large
      q(large) = q(large) + q(small) - 1.0
      if (q(large) < 1.0) smaller.append(large)
      else larger.append(large)
    }

    (J, q)
  }

  def setupEdgeAlias(p: Double = 1.0, q: Double = 1.0)(srcId: Long, srcNeighbors: Array[(Long, Double)], dstNeighbors: Array[(Long, Double)]): (Array[Int], Array[Double]) = {
    val neighbors_ = dstNeighbors.map { case (dstNeighborId, weight) =>
      var unnormProb = weight / q
      if (srcId == dstNeighborId) unnormProb = weight / p
      else if (srcNeighbors.exists(_._1 == dstNeighborId)) unnormProb = weight

      (dstNeighborId, unnormProb)
    }

    setupAlias(neighbors_)
  }

  def drawAlias(J: Array[Int], q: Array[Double]): Int = {
    val K = J.length
    val kk = math.floor(math.random * K).toInt

    if (math.random < q(kk)) kk
    else J(kk)
  }

  lazy val createUndirectedEdge = (srcId: Long, dstId: Long, weight: Double) => {
    Array(
      (srcId, Array((dstId, weight))),
      (dstId, Array((srcId, weight)))
    )
  }
  
  lazy val createDirectedEdge = (srcId: Long, dstId: Long, weight: Double) => {
    Array(
      (srcId, Array((dstId, weight)))
    )
  }
}
